# -*- coding: utf-8 -*-
"""
视觉-语言-动作(VLA)模型训练脚本
用于机器人行为克隆任务的端到端训练

主要功能：
1. 配置和管理训练参数
2. 加载和处理机器人演示数据
3. 训练VLA模型进行动作预测
4. 支持LoRA微调和分布式训练
"""

import pickle
import os
import time

# 环境变量设置
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行化避免死锁
os.environ['DEVICE'] = "cuda"                   # 设置GPU设备
os.environ["WANDB_DISABLED"] = "true"           # 禁用Weights & Biases日志记录

import torch
from policy_heads import *                       # 导入策略头模块(动作预测网络)
from data_utils.dataset import set_seed, load_data  # 数据加载和种子设置工具

from vla import *                               # VLA模型核心模块
from aloha_scripts.utils import *               # ALOHA机器人工具函数
from aloha_scripts.constants import TASK_CONFIGS  # 任务配置常量
from transformers import AutoConfig, AutoProcessor, AutoTokenizer  # Transformers库组件
from data_utils.data_collator import DataCollatorForSupervisedDataset  # 数据整理器
from data_utils.robot_data_processor import InternVL3Process  # 机器人数据处理器
from dataclasses import dataclass, field, asdict  # 数据类装饰器

# 全局变量，用于分布式训练的rank管理
local_rank = None


def rank0_print(*args):
    """
    只在rank 0进程中打印信息的工具函数
    用于分布式训练时避免重复打印
    """
    if local_rank == 0:
        print(*args)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 参数配置类定义 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

@dataclass
class ActionHeadArguments:
    """
    动作头(策略网络)相关参数配置
    定义了动作预测网络的结构和输入输出维度
    """
    policy_head_type: str = field(default="unet_diffusion_policy")  # 策略头类型：UNet扩散策略
    state_dim: int = 7          # 状态维度(如机器人关节角度等)
    action_dim: int = 10        # 动作维度(如机器人控制指令维度)
    noise_samples: int = 1      # 扩散模型的噪声采样数量


@dataclass
class ModelArguments:
    """
    模型相关参数配置
    定义了基础语言模型的路径和优化设置
    """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")  # 预训练模型路径
    flash_attn: bool = field(default=False)  # 是否使用Flash Attention优化


@dataclass
class DataArguments:
    """
    数据相关参数配置
    定义了数据加载、处理和任务选择的参数
    """
    episode_first: bool = False                        # 是否按episode优先排序
    task_name: str = field(default="stack_cube_2024_6_2")  # 任务名称
    skip_mirrored_data: bool = field(default=False)    # 是否跳过镜像数据增强
    chunk_size: int = field(default=16)                # 数据块大小(序列长度)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    训练相关参数配置
    继承transformers的TrainingArguments并添加自定义参数
    """
    local_debug: bool = field(default=False)  # 本地调试模式

    # 缓存和优化器设置
    cache_dir: Optional[str] = field(default=None)     # 缓存目录
    optim: str = field(default="adamw_torch")          # 优化器类型
    adam_beta1: float = field(default=0.9)             # Adam优化器beta1参数
    adam_beta2: float = field(default=0.98)            # Adam优化器beta2参数
    adam_epsilon: float = field(default=1e-7)          # Adam优化器epsilon参数
    seed: int = field(default=0)                       # 随机种子

    # 模型冻结设置
    freeze_vision_tower: bool = field(default=False)   # 是否冻结视觉编码器
    freeze_backbone: bool = field(default=False)       # 是否冻结主干网络

    # 日志记录设置
    logging_dir: str = field(default='./logs')         # 日志目录
    logging_strategy: str = field(default='steps')     # 日志记录策略
    logging_steps: int = field(default=10)             # 日志记录步数间隔

    # 模型保存设置
    save_steps: int = field(default=10)                # 模型保存步数间隔
    max_steps: int = field(default=10000)              # 最大训练步数

    # 数据加载优化
    dataloader_pin_memory: bool = True                 # 是否将数据固定在内存中

    # LoRA (Low-Rank Adaptation) 微调设置
    lora_enable: bool = False                          # 是否启用LoRA微调
    lora_module: str = "vit"                           # LoRA应用的模块
    lora_task_type: str = 'CAUSAL_LM'                  # LoRA任务类型
    lora_r: int = 64                                   # LoRA秩大小
    lora_alpha: int = 256                              # LoRA缩放参数
    lora_dropout: float = 0.05                         # LoRA dropout率
    lora_weight_path: str = ""                         # LoRA权重路径
    lora_bias: str = "none"                            # LoRA偏置设置
    policy_head_lr: Optional[float] = None             # 策略头学习率

    # 序列长度和精度设置
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}  # 量化位数
    )


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 参数配置类定义结束 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def parse_param():
    """
    解析命令行参数和配置文件
    
    Returns:
        tuple: 包含模型参数、数据参数、训练参数、动作头参数和模型配置的元组
    """
    global local_rank

    # 使用HuggingFace的参数解析器解析所有参数类
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments)
    )
    model_args, data_args, training_args, action_head_args = parser.parse_args_into_dataclasses()
    
    # 设置分布式训练的local rank
    local_rank = training_args.local_rank
    
    # 从预训练模型加载配置，并添加动作头参数
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=False, 
        **asdict(action_head_args)
    )

    # 根据策略头类型配置相应的网络结构
    cond_dim = config.hidden_size  # 条件维度(来自语言模型的隐藏状态维度)
    
    if action_head_args.policy_head_type == 'unet_diffusion_policy':
        # 配置UNet扩散策略网络
        config.policy_head_config = AutoConfig.for_model(
            model_type=config.policy_head_type,
            global_cond_dim=cond_dim,                    # 全局条件维度
            action_dim=action_head_args.action_dim,      # 动作空间维度
            state_dim=action_head_args.state_dim,        # 状态空间维度
            noise_samples=action_head_args.noise_samples, # 噪声采样数
        )
    else:
        raise NotImplementedError(f"Unsupported policy head type {action_head_args.policy_head_type}")

    # 将模型参数添加到配置中
    for k, v in asdict(model_args).items():
        setattr(config, k, v)

    return model_args, data_args, training_args, action_head_args, config


def train_bc(train_dataset=None, model=None, config=None, tokenizer=None):
    """
    执行行为克隆训练的核心函数
    
    Args:
        train_dataset: 训练数据集
        model: 要训练的VLA模型
        config: 包含所有配置参数的字典
        tokenizer: 文本tokenizer
    """
    # 设置随机种子确保可重现性
    set_seed(config['training_args'].seed)
    
    # 根据训练参数确定计算数据类型(fp16/bf16/fp32)
    compute_dtype = (
        torch.float16 if training_args.fp16 else 
        (torch.bfloat16 if config['training_args'].bf16 else torch.float32)
    )
    
    # 创建数据整理器，用于批处理数据
    data_collator = DataCollatorForSupervisedDataset(
        computed_type=compute_dtype, 
        tokenizer=tokenizer
    )

    # 启用模型缓存以提高推理速度
    model.config.use_cache = True
    
    # 确保策略头配置为字典格式并保存
    if not isinstance(model.config.policy_head_config, dict):
        model.config.policy_head_config = model.config.policy_head_config.to_dict()
    model.config.save_pretrained(config['training_args'].output_dir)
    
    # 准备训练数据模块
    data_module = dict(
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # 创建VLA训练器
    trainer = VLATrainer(
        model=model,
        tokenizer=tokenizer,
        args=config['training_args'],
        **data_module
    )

    # 开始训练(支持从检查点恢复)
    trainer.train(resume_from_checkpoint=config['training_args'].resume_from_checkpoint)

    # 保存训练状态
    trainer.save_state()

    # 重新启用缓存
    model.config.use_cache = True

    # 根据是否使用LoRA采用不同的保存策略
    if config['training_args'].lora_enable:
        # LoRA模式：分别保存LoRA权重和非LoRA权重
        state_dict = model_load_utils.get_peft_state_maybe_zero_3(
            model.named_parameters(), config['training_args'].lora_bias
        )
        non_lora_state_dict = model_load_utils.get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        
        # 只在主进程中保存模型
        if config['training_args'].local_rank == 0 or config['training_args'].local_rank == -1:
            model.config.save_pretrained(config['training_args'].output_dir)
            model.save_pretrained(config['training_args'].output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(config['training_args'].output_dir, 'non_lora_trainables.bin')
            )
    else:
        # 标准模式：保存完整模型
        model_load_utils.safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=config['training_args'].output_dir
        )


def main(all_config, model_config):
    """
    主训练流程函数
    
    Args:
        all_config: 包含所有配置参数的字典
        model_config: 模型特定配置
    """
    # 设置随机种子
    set_seed(all_config["training_args"].seed)

    # 根据任务名称获取任务配置(摄像头、数据路径等)
    task_config = TASK_CONFIGS[all_config['data_args'].task_name]
    camera_names = task_config['camera_names']  # 摄像头名称列表
    dataset_dir = task_config['dataset_dir']    # 数据集目录

    # 将摄像头配置添加到模型配置中
    model_config.camera_names = task_config['camera_names']
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        all_config['model_args'].model_name_or_path,
    )
    
    # 加载VLA模型
    model, data_args = model_load_utils.load_model(
        config=all_config, 
        vla_config=model_config, 
        rank0_print=rank0_print
    )

    rank0_print(f"{RED} Using {all_config['model_args'].model_name_or_path} as VLA backbone {RESET}")
    
    # 创建VLA数据处理器(处理图像和文本输入)
    vla_process = InternVL3Process(
        tokenizer=tokenizer,
        conv_template=model.conv_template,      # 对话模板
        data_args=all_config['data_args'],
        camera_names=camera_names,
        num_image_token=model.num_image_token   # 图像token数量
    )

    # 加载训练数据集和统计信息
    train_dataset, stats = load_data(
        dataset_dir_l=dataset_dir,
        skip_mirrored_data=all_config['data_args'].skip_mirrored_data,
        camera_names=camera_names,
        chunk_size=all_config['data_args'].chunk_size,
        config=all_config,
        rank0_print=rank0_print,
        policy_class=all_config['action_head_args'].policy_head_type,
        vla_data_post_process=vla_process
    )

    # 保存数据集统计信息
    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # 执行行为克隆训练
    train_bc(
        train_dataset=train_dataset,
        model=model,
        config=all_config,
        tokenizer=tokenizer
    )
    
    # 再次保存数据集统计信息(确保训练完成后信息完整)
    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)


if __name__ == '__main__':
    """
    程序入口点
    1. 解析命令行参数
    2. 组织配置字典
    3. 检查是否从检查点恢复训练
    4. 启动主训练流程
    """
    # 解析所有配置参数
    model_args, data_args, training_args, action_head_args, model_config = parse_param()
    
    # 组织配置字典
    config = {
        'model_args': model_args,
        'data_args': data_args,
        'training_args': training_args,
        'action_head_args': action_head_args,
    }

    # 将数据类转换为字典格式
    config_dict = {
        k: asdict(v) if not isinstance(v, dict) else v 
        for k, v in config.items()
    }

    # 检查输出目录中的检查点
    ckpt = os.listdir(config['training_args'].output_dir)
    if config['training_args'].resume_from_checkpoint is not None:
        rank0_print(f"{RED}Resuming Training from {config['training_args'].resume_from_checkpoint}............{RESET}")
    
    # 启动主训练流程
    main(all_config=config, model_config=model_config)
