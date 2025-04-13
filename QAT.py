import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入 QAT 相关模块
from torch.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qat_qconfig, prepare_qat, convert

class ConvBNReLUQAT(nn.Module):
    """
    一个包含 Conv2d -> BatchNorm2d -> ReLU 的模块，
    并集成了 QAT 所需的 QuantStub 和 DeQuantStub。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvBNReLUQAT, self).__init__()
        # 为了 BatchNorm 能被融合，Conv2d 的 bias 通常设置为 False
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # QuantStub: 标记量化的入口点，将浮点输入转换为量化格式
        self.quant = QuantStub()
        # DeQuantStub: 标记量化的出口点，将量化格式转换回浮点输出
        self.dequant = DeQuantStub()

    def forward(self, x):
        # 1. 量化输入
        x = self.quant(x)
        # 2. 在 QAT 训练期间，Conv, BN, ReLU 顺序执行
        #    PyTorch 会在它们之间插入观察者 (Observers) 来收集统计信息
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 3. 反量化输出
        x = self.dequant(x)
        return x

    def fuse_model_manually(self):
        """
        演示如何手动融合普通（非量化）模型中的 Conv-BN-ReLU。
        注意：在标准的 QAT 流程中，这个方法不需要被调用，
              融合是由 torch.quantization.convert() 自动完成的。
        """
        print("Attempting manual fusion on the original model structure...")
        # 使用 PyTorch 提供的 fuse_modules 进行融合
        fuse_modules(self, ['conv', 'bn', 'relu'], inplace=True)
        print("Manual fusion complete (if applicable).")


# --- 示例用法和 QAT 流程 ---
if __name__ == '__main__':
    # 1. 创建模型实例
    model = ConvBNReLUQAT(in_channels=3, out_channels=64, kernel_size=3, padding=1)

    # 2. QAT 设置
    model.train() # QAT 需要在训练模式下进行准备
    # 选择 QAT 配置 (决定了使用哪种观察者和量化方案)
    # 'fbgemm' 是推荐用于 x86 CPU 的后端
    model.qconfig = get_default_qat_qconfig('fbgemm')
    print("--- Original Model Structure ---")
    print(model)
    print("\n--- Preparing Model for QAT ---")
    # 准备模型进行 QAT: 插入观察者 (Observers)
    model_prepared = prepare_qat(model)
    print("--- Model Prepared for QAT (with Observers) ---")
    print(model_prepared)

    # 3. 模拟训练过程
    #    在实际应用中，这里会用真实数据进行多个周期的训练
    #    训练的目的是让模型适应模拟量化的影响，并让观察者收集数据范围信息
    print("\n--- Starting Dummy Training ---")
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    num_epochs = 1 # 仅作演示
    # 创建一些虚拟数据
    dataloader = [(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))) for _ in range(5)]
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            # 使用虚拟损失
            loss = F.cross_entropy(outputs.mean(dim=[2,3]), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    print("--- Dummy Training Complete ---")

    # 4. 转换到最终的量化模型 (包含自动融合)
    model_prepared.eval() # 转换前需要设置为评估模式
    # 这是关键步骤：
    # - 使用观察者收集到的信息计算量化参数 (scale, zero_point)
    # - 自动识别可融合的模式 (如 Conv-BN-ReLU)
    # - 执行算子融合 (将 BN 参数折叠进 Conv)
    # - 用融合后的量化层替换原始层序列
    model_quantized = convert(model_prepared)
    print("\n--- Model Converted to Quantized (with Fusion) ---")
    # 观察打印出的结构，Conv, BN, ReLU 会被替换为类似 QuantizedConvReLU2d 的单一模块
    print(model_quantized)

    # 5. (可选) 对比手动融合原始模型
    print("\n--- Manually Fusing Original Model (for Comparison) ---")
    # 创建一个新的原始模型实例
    model_orig_for_fusion = ConvBNReLUQAT(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    model_orig_for_fusion.eval()
    # 调用我们之前定义的 手动融合 方法
    model_orig_for_fusion.fuse_model_manually()
    print("--- Original Model Structure After Manual Fusion ---")
    print(model_orig_for_fusion)

    # 6. (可选) 验证量化模型的输出 (需要输入数据)
    print("\n--- Testing Quantized Model Output ---")
    dummy_input = torch.randn(1, 3, 32, 32)
    output_quantized = model_quantized(dummy_input)
    print(f"Output shape from quantized model: {output_quantized.shape}")
