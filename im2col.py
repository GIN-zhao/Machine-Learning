import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    """计算 im2col 所需的索引"""
    N, C, H, W = x_shape
    # Allow non-integer results for calculation before floor division
    out_height_float = (H + 2 * padding - field_height) / stride + 1
    out_width_float = (W + 2 * padding - field_width) / stride + 1

    # Check if output dimensions are integers before floor division
    # Use a small tolerance for floating point comparisons
    tolerance = 1e-6
    if not (abs(out_height_float - np.round(out_height_float)) < tolerance and \
            abs(out_width_float - np.round(out_width_float)) < tolerance):
         raise ValueError("Output dimensions must be integers. Check input shape, kernel size, padding, and stride.")

    out_height = int(np.round(out_height_float))
    out_width = int(np.round(out_width_float))


    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C) # (C*KH*KW,)
    i1 = stride * np.repeat(np.arange(out_height), out_width) # (OH*OW,)
    j0 = np.tile(np.arange(field_width), field_height * C) # (C*KH*KW,)
    j1 = stride * np.tile(np.arange(out_width), out_height) # (OH*OW,)

    # i = i0 + i1: Shape (C*KH*KW, OH*OW) -> broadcast i0 and i1
    # j = j0 + j1: Shape (C*KH*KW, OH*OW) -> broadcast j0 and j1
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    # k corresponds to the channel index
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1) # (C*KH*KW, 1)

    return (k, i, j) # Indices for C, H, W dimensions

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    im2col 的一种实现方式，使用预先计算的索引。
    输入 x: 形状为 (N, C, H, W) 的图像数据
    输出: 形状为 (C * field_height * field_width, N * out_height * out_width) 的矩阵
    """
    # Zero-pad the input
    p = padding
    # Ensure padding is integer if it's a single number
    if isinstance(padding, (int, float)):
        p = int(padding)
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
         # Assuming (pad_h, pad_w)
         ph, pw = int(padding[0]), int(padding[1])
         x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant')
    else:
        raise TypeError("Padding must be an integer or a tuple/list of two integers (pad_h, pad_w)")


    # Ensure stride is integer or tuple/list of two integers
    if isinstance(stride, (int, float)):
        stride_h = stride_w = int(stride)
    elif isinstance(stride, (tuple, list)) and len(stride) == 2:
        stride_h, stride_w = int(stride[0]), int(stride[1])
    else:
        raise TypeError("Stride must be an integer or a tuple/list of two integers (stride_h, stride_w)")


    k, i, j = get_im2col_indices(x.shape, field_height, field_width, p, stride_h) # Use stride_h for index calc

    # Get the columns using advanced indexing
    # cols shape: (N, C*KH*KW, OH*OW) after indexing x_padded[np.arange(N)[:, None, None], k, i, j]
    # Need to handle batch dimension N correctly during indexing
    N = x.shape[0]
    cols = x_padded[np.arange(N)[:, None, None], k, i, j] # Shape (N, C*KH*KW, OH*OW)

    # Reshape and transpose to match the desired output format
    # (N, C*KH*KW, OH*OW) -> transpose -> (C*KH*KW, N, OH*OW) -> reshape -> (C*KH*KW, N*OH*OW)
    C = x.shape[1]
    out_height = (x.shape[2] + 2 * (ph if isinstance(p, tuple) else p) - field_height) // stride_h + 1
    out_width = (x.shape[3] + 2 * (pw if isinstance(p, tuple) else p) - field_width) // stride_w + 1 # Use stride_w here

    cols = cols.transpose(1, 0, 2).reshape(C * field_height * field_width, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    """im2col 的逆运算 (使用循环实现，可能较慢但更清晰)"""
    N, C, H, W = x_shape

    # Ensure padding is integer or tuple/list
    if isinstance(padding, (int, float)):
        pad_h = pad_w = int(padding)
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        pad_h, pad_w = int(padding[0]), int(padding[1])
    else:
        raise TypeError("Padding must be an integer or a tuple/list of two integers (pad_h, pad_w)")

    # Ensure stride is integer or tuple/list
    if isinstance(stride, (int, float)):
        stride_h = stride_w = int(stride)
    elif isinstance(stride, (tuple, list)) and len(stride) == 2:
        stride_h, stride_w = int(stride[0]), int(stride[1])
    else:
        raise TypeError("Stride must be an integer or a tuple/list of two integers (stride_h, stride_w)")


    H_padded, W_padded = H + 2 * pad_h, W + 2 * pad_w
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    out_h = (H + 2 * pad_h - field_height) // stride_h + 1
    out_w = (W + 2 * pad_w - field_width) // stride_w + 1

    # Reshape cols: (C*KH*KW, N*OH*OW) -> (C, KH, KW, N, OH, OW)
    cols_reshaped = cols.reshape(C, field_height, field_width, N, out_h, out_w)

    for n in range(N):
        for c in range(C):
            for kh in range(field_height):
                for kw in range(field_width):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_pad = h * stride_h + kh
                            w_pad = w * stride_w + kw
                            # Accumulate gradients/values for overlapping regions
                            x_padded[n, c, h_pad, w_pad] += cols_reshaped[c, kh, kw, n, h, w]

    # Remove padding
    if pad_h == 0 and pad_w == 0:
        return x_padded
    elif pad_h > 0 and pad_w > 0:
         return x_padded[:, :, pad_h:-pad_h, pad_w:-pad_w]
    elif pad_h > 0: # Only height padding
         return x_padded[:, :, pad_h:-pad_h, :]
    else: # Only width padding
         return x_padded[:, :, :, pad_w:-pad_w]


# --- 示例用法 ---
if __name__ == '__main__':
    # 创建一个简单的输入图像 (Batch=1, Channel=1, Height=4, Width=4)
    x = np.arange(16).reshape(1, 1, 4, 4)
    print("Input image (x):")
    print(x)
    print("Shape:", x.shape)

    # 定义卷积参数
    kernel_h, kernel_w = 3, 3
    stride = 1 # stride_h=1, stride_w=1
    padding = 1 # pad_h=1, pad_w=1

    # 执行 im2col
    cols = im2col_indices(x, kernel_h, kernel_w, padding=padding, stride=stride)
    print("\nOutput of im2col (cols):")
    print(cols)
    print("Shape:", cols.shape) # Expected: (C*KH*KW, N*OH*OW) = (1*3*3, 1*4*4) = (9, 16)

    # 验证输出尺寸
    N, C, H, W = x.shape
    OH = (H + 2 * padding - kernel_h) // stride + 1
    OW = (W + 2 * padding - kernel_w) // stride + 1
    print(f"\nExpected output height (OH): {OH}")
    print(f"Expected output width (OW): {OW}")
    expected_cols_shape = (C * kernel_h * kernel_w, N * OH * OW)
    print(f"Expected cols shape: {expected_cols_shape}")
    assert cols.shape == expected_cols_shape, f"Shape mismatch: {cols.shape} vs {expected_cols_shape}"

    # --- 添加验证 ---
    print("\n--- Verifying im2col against naive convolution ---")
    # 定义一个卷积核 (Cout=2, Cin=1, KH=3, KW=3)
    kernel = np.arange(18).reshape(2, 1, 3, 3) # Cout=2, Cin=C, KH, KW
    Cout = kernel.shape[0]

    # 1. 使用 im2col 和矩阵乘法计算卷积
    kernel_reshaped = kernel.reshape(Cout, -1) # (Cout, Cin*KH*KW)
    conv_out_im2col_flat = kernel_reshaped @ cols # (Cout, N*OH*OW)
    # 将结果 reshape 回图像格式 (N, Cout, OH, OW)
    conv_out_im2col = conv_out_im2col_flat.reshape(Cout, N, OH, OW).transpose(1, 0, 2, 3) # (N, Cout, OH, OW)

    # 2. 使用朴素卷积循环计算结果 (作为参考)
    def naive_conv2d(input_x, kernel_w, stride_s=1, padding_p=1):
        N_in, C_in, H_in, W_in = input_x.shape
        C_out, C_in_k, KH, KW = kernel_w.shape
        assert C_in == C_in_k, "Input channels must match kernel channels"

        # Apply padding
        if padding_p > 0:
            x_pad = np.pad(input_x, ((0, 0), (0, 0), (padding_p, padding_p), (padding_p, padding_p)), mode='constant')
        else:
            x_pad = input_x

        # Calculate output dimensions
        H_out = (H_in + 2 * padding_p - KH) // stride_s + 1
        W_out = (W_in + 2 * padding_p - KW) // stride_s + 1
        output = np.zeros((N_in, C_out, H_out, W_out))

        for n in range(N_in):           # Batch
            for c_out in range(C_out):   # Output channels
                for h_out in range(H_out): # Output height
                    for w_out in range(W_out): # Output width
                        # Extract the receptive field
                        h_start = h_out * stride_s
                        h_end = h_start + KH
                        w_start = w_out * stride_s
                        w_end = w_start + KW
                        receptive_field = x_pad[n, :, h_start:h_end, w_start:w_end]
                        # Perform element-wise multiplication and sum
                        output[n, c_out, h_out, w_out] = np.sum(receptive_field * kernel_w[c_out])
        return output

    conv_out_naive = naive_conv2d(x, kernel, stride_s=stride, padding_p=padding)

    print("Shape of output (im2col + matmul):", conv_out_im2col.shape)
    print("Shape of output (naive conv):", conv_out_naive.shape)
    assert conv_out_im2col.shape == conv_out_naive.shape, "Output shapes do not match"

    # 3. 比较结果
    print("Are im2col and naive convolution results close?", np.allclose(conv_out_im2col, conv_out_naive))
    if not np.allclose(conv_out_im2col, conv_out_naive):
        print("Difference:", np.abs(conv_out_im2col - conv_out_naive).max())


    # (可选) 执行 col2im
    print("\nAttempting col2im (Note: col2im validation is separate)...")
    try:
        x_reconstructed = col2im_indices(cols, x.shape, kernel_h, kernel_w, padding=padding, stride=stride)
        print("Reconstructed image shape:", x_reconstructed.shape)
        print("Reconstructed image (x_reconstructed):")
        print(x_reconstructed)
        # 对于非重叠或简单情况，可能接近原图
        print("Is reconstructed close to original?", np.allclose(x, x_reconstructed))
    except Exception as e:
        print(f"col2im execution failed or needs debugging: {e}")

    print("\n--- Example with different parameters (Modified Input Size) ---")
    # Batch=2, Channel=3, Height=6, Width=6 (Changed from 5x5 to make parameters valid)
    x2 = np.random.randn(2, 3, 6, 6)
    kernel_h2, kernel_w2 = 2, 2
    stride2 = 2 # stride_h=2, stride_w=2
    padding2 = 0 # No padding

    cols2 = im2col_indices(x2, kernel_h2, kernel_w2, padding=padding2, stride=stride2)
    N2, C2, H2, W2 = x2.shape
    OH2 = (H2 + 2 * padding2 - kernel_h2) // stride2 + 1
    OW2 = (W2 + 2 * padding2 - kernel_w2) // stride2 + 1
    expected_cols_shape2 = (C2 * kernel_h2 * kernel_w2, N2 * OH2 * OW2)
    print(f"Input shape: {x2.shape}")
    print(f"Kernel: ({kernel_h2}, {kernel_w2}), Stride: {stride2}, Padding: {padding2}")
    print(f"Output Height/Width: ({OH2}, {OW2})")
    print(f"im2col output shape: {cols2.shape}")
    print(f"Expected im2col shape: {expected_cols_shape2}")
    assert cols2.shape == expected_cols_shape2, f"Shape mismatch: {cols2.shape} vs {expected_cols_shape2}"

    # --- 添加验证 (第二个示例) ---
    print("\n--- Verifying im2col against naive convolution (Example 2) ---")
    # 定义卷积核 (Cout=4, Cin=3, KH=2, KW=2)
    kernel2 = np.random.randn(4, 3, 2, 2) # Cout=4, Cin=C2, KH2, KW2
    Cout2 = kernel2.shape[0]

    # 1. 使用 im2col 和矩阵乘法
    kernel2_reshaped = kernel2.reshape(Cout2, -1) # (Cout2, Cin2*KH2*KW2)
    conv_out_im2col_flat2 = kernel2_reshaped @ cols2 # (Cout2, N2*OH2*OW2)
    conv_out_im2col2 = conv_out_im2col_flat2.reshape(Cout2, N2, OH2, OW2).transpose(1, 0, 2, 3) # (N2, Cout2, OH2, OW2)

    # 2. 使用朴素卷积循环
    conv_out_naive2 = naive_conv2d(x2, kernel2, stride_s=stride2, padding_p=padding2)

    print("Shape of output (im2col + matmul):", conv_out_im2col2.shape)
    print("Shape of output (naive conv):", conv_out_naive2.shape)
    assert conv_out_im2col2.shape == conv_out_naive2.shape, "Output shapes do not match"

    # 3. 比较结果
    print("Are im2col and naive convolution results close?", np.allclose(conv_out_im2col2, conv_out_naive2))
    if not np.allclose(conv_out_im2col2, conv_out_naive2):
        print("Difference:", np.abs(conv_out_im2col2 - conv_out_naive2).max())


    # Test col2im for the second example
    print("\nAttempting col2im for second example (Note: col2im validation is separate)...")
    try:
        x2_reconstructed = col2im_indices(cols2, x2.shape, kernel_h2, kernel_w2, padding=padding2, stride=stride2)
        print("Reconstructed image shape:", x2_reconstructed.shape)
        # In non-overlapping cases, reconstruction might be easier to verify,
        # but col2im still sums contributions if stride < kernel size.
        # Since stride=kernel size here, it should be non-overlapping.
        # However, the original image cannot be perfectly reconstructed from cols
        # if information was 'lost' due to striding > 1 without overlap.
        # col2im essentially places the columns back into an image grid.
        print("Reconstructed image (x2_reconstructed, first batch/channel):")
        print(x2_reconstructed[0, 0, :, :])
    except Exception as e:
        print(f"col2im execution failed or needs debugging: {e}")
