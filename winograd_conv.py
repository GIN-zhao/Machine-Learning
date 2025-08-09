import numpy as np

# --- Winograd F(2x2, 3x3) Transformation Matrices ---
# See paper: "Fast Algorithms for Convolutional Neural Networks" by Lavin and Gray (https://arxiv.org/abs/1509.09308)

# Kernel transformation matrix G (for 3x3 kernel -> 4x4 Winograd domain)
G = np.array([
    [1,  0,  0],
    [0.5, 0.5, 0.5],
    [0.5,-0.5, 0.5],
    [0,  0,  1]
], dtype=np.float32)

# Input transformation matrix B^T (for 4x4 input tile -> 4x4 Winograd domain)
BT = np.array([
    [ 1,  0, -1,  0],
    [ 0,  1,  1,  0],
    [ 0, -1,  1,  0],
    [ 0,  1,  0, -1]
], dtype=np.float32)
B = BT.T

# Output transformation matrix A^T (for 4x4 Winograd domain result -> 2x2 output tile)
AT = np.array([
    [1, 1,  1, 0],
    [0, 1, -1, -1]
], dtype=np.float32)
A = AT.T

def transform_kernel(kernel):
    """Transforms the kernel (g) into the Winograd domain (U).
       kernel shape: (Cout, Cin, 3, 3)
       output shape: (Cout, Cin, 4, 4)
    """
    Cout, Cin, KH, KW = kernel.shape
    assert KH == 3 and KW == 3, "Winograd F(2x2, 3x3) requires a 3x3 kernel."
    # U = G @ g @ G.T
    # Apply G row-wise, then G.T column-wise using einsum or matmul
    # G @ g -> (Cout, Cin, 4, 3)
    # (G @ g) @ G.T -> (Cout, Cin, 4, 4)
    # Use np.tensordot for batch matrix multiplication across channels
    # G (4,3), kernel (O,I,3,3) -> temp (O,I,4,3)
    temp = np.tensordot(G, kernel, axes=([1],[2])) # Sum over KH axis
    # Need to rearrange axes for the second multiplication
    # temp (O,I,4,3), G.T (3,4) -> U (O,I,4,4)
    # We want to multiply the last axis of temp with the first axis of G.T
    U = np.tensordot(temp, G.T, axes=([3],[0])) # Sum over KW axis
    return U

def transform_input_tile(tile):
    """Transforms a 4x4 input tile (d) into the Winograd domain (V).
       tile shape: (Cin, 4, 4) or (N, Cin, 4, 4) - Assuming (Cin, 4, 4) here
       output shape: (Cin, 4, 4)
    """
    # V = B.T @ d @ B
    # Apply B.T row-wise, then B column-wise
    # BT (4,4), tile (I,4,4) -> temp (I,4,4)
    temp = np.tensordot(BT, tile, axes=([1],[1])) # Sum over H axis
    # temp (I,4,4), B (4,4) -> V (I,4,4)
    V = np.tensordot(temp, B, axes=([2],[0])) # Sum over W axis
    return V

def inverse_transform_output_tile(m):
    """Transforms a 4x4 Winograd domain result (M) back to a 2x2 output tile (Y).
       m shape: (Cout, 4, 4)
       output shape: (Cout, 2, 2)
    """
    # Y = A.T @ M @ A
    # Apply A.T row-wise, then A column-wise
    # AT (2,4), m (O,4,4) -> temp (O,2,4)
    temp = np.tensordot(AT, m, axes=([1],[1])) # Sum over first spatial dim
    # temp (O,2,4), A (4,2) -> Y (O,2,2)
    Y = np.tensordot(temp, A, axes=([2],[0])) # Sum over second spatial dim
    return Y

def winograd_conv2d_f2x2_3x3(x, kernel, padding=1, stride=1):
    """
    Performs 2D convolution using Winograd F(2x2, 3x3).
    Assumes stride = 1 and padding allows tiling without remainder.
    Input x: shape (N, Cin, H, W)
    Kernel kernel: shape (Cout, Cin, 3, 3)
    Output: shape (N, Cout, H_out, W_out)
    """
    if stride != 1:
        raise NotImplementedError("Winograd F(2x2, 3x3) implemented here assumes stride=1.")

    N, Cin, H, W = x.shape
    Cout, Cin_k, KH, KW = kernel.shape
    assert Cin == Cin_k, "Input channels must match kernel channels."
    assert KH == 3 and KW == 3, "Kernel must be 3x3."

    # Apply padding
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        x_padded = x
    N_pad, Cin_pad, H_pad, W_pad = x_padded.shape # Use different names to avoid confusion

    # Calculate output dimensions (for stride=1, padding=1, kernel=3x3 -> H_out=H, W_out=W)
    H_out = H + 2 * padding - KH + 1
    W_out = W + 2 * padding - KW + 1
    output = np.zeros((N, Cout, H_out, W_out), dtype=x.dtype)

    # Output tile size is 2x2, Input tile size is 4x4
    output_tile_size = 2
    input_tile_size = 4

    # Check if dimensions are suitable for tiling
    if (H_pad - input_tile_size) % output_tile_size != 0 or \
       (W_pad - input_tile_size) % output_tile_size != 0:
        # This implementation requires exact tiling for simplicity.
        # More advanced implementations handle remainders.
        raise ValueError(f"Input dimensions after padding ({H_pad}x{W_pad}) are not suitable for F(2x2, 3x3) tiling with input tile size {input_tile_size} and output tile size {output_tile_size}.")

    num_tiles_h = (H_pad - input_tile_size) // output_tile_size + 1
    num_tiles_w = (W_pad - input_tile_size) // output_tile_size + 1

    # 1. Transform Kernel
    U = transform_kernel(kernel) # Shape: (Cout, Cin, 4, 4)

    # 2. Iterate through tiles
    for n in range(N):
        for tile_r in range(num_tiles_h):
            for tile_c in range(num_tiles_w):
                # Calculate input tile coordinates
                r_start = tile_r * output_tile_size
                r_end = r_start + input_tile_size
                c_start = tile_c * output_tile_size
                c_end = c_start + input_tile_size

                # Extract input tile
                input_tile = x_padded[n, :, r_start:r_end, c_start:c_end] # Shape: (Cin, 4, 4)

                # 3. Transform Input Tile
                V = transform_input_tile(input_tile) # Shape: (Cin, 4, 4)

                # 4. Element-wise Multiplication in Winograd Domain
                # M_cin = U_cout_cin * V_cin (element-wise)
                # Result M needs shape (Cout, 4, 4) after summing over Cin
                # Use einsum for clarity: 'oikm,ikm->okm'
                # o=Cout, i=Cin, k=4, m=4
                M = np.einsum('oikm,ikm->okm', U, V, optimize=True) # Shape: (Cout, 4, 4)

                # 5. Inverse Transform Output Tile
                output_tile = inverse_transform_output_tile(M) # Shape: (Cout, 2, 2)

                # 6. Place output tile into the final output tensor
                out_r_start = tile_r * output_tile_size
                out_r_end = out_r_start + output_tile_size
                out_c_start = tile_c * output_tile_size
                out_c_end = out_c_start + output_tile_size

                # Ensure indices are within bounds of the calculated output size
                if out_r_end <= H_out and out_c_end <= W_out:
                     output[n, :, out_r_start:out_r_end, out_c_start:out_c_end] = output_tile
                # Handle boundary cases if output size is not perfectly divisible by tile size
                # (This basic implementation assumes perfect divisibility based on the check above)

    return output

# --- Helper function for naive convolution (from im2col.py for verification) ---
def naive_conv2d(input_x, kernel_w, stride_s=1, padding_p=1):
    N_in, C_in, H_in, W_in = input_x.shape
    C_out, C_in_k, KH, KW = kernel_w.shape
    assert C_in == C_in_k, "Input channels must match kernel channels"

    # Apply padding
    if padding_p > 0:
        x_pad = np.pad(input_x, ((0, 0), (0, 0), (padding_p, padding_p), (padding_p, padding_p)), mode='constant')
    else:
        x_pad = input_x
    N_pad, C_pad, H_pad, W_pad = x_pad.shape # Get padded dimensions

    # Calculate output dimensions
    H_out = (H_in + 2 * padding_p - KH) // stride_s + 1
    W_out = (W_in + 2 * padding_p - KW) // stride_s + 1
    output = np.zeros((N_in, C_out, H_out, W_out), dtype=input_x.dtype) # Match dtype

    for n in range(N_in):           # Batch
        for c_out in range(C_out):   # Output channels
            for h_out in range(H_out): # Output height
                for w_out in range(W_out): # Output width
                    # Extract the receptive field
                    h_start = h_out * stride_s
                    h_end = h_start + KH
                    w_start = w_out * stride_s
                    w_end = w_start + KW
                    # Ensure receptive field indices are within padded bounds
                    receptive_field = x_pad[n, :, h_start:h_end, w_start:w_end]
                    # Perform element-wise multiplication and sum
                    output[n, c_out, h_out, w_out] = np.sum(receptive_field * kernel_w[c_out])
    return output

# --- Example Usage & Verification ---
if __name__ == '__main__':
    # Define input and kernel
    # Input size must be suitable for tiling (e.g., 6x6 padded becomes 8x8)
    # Padded H/W must satisfy (H_pad - 4) % 2 == 0 and (W_pad - 4) % 2 == 0
    N, Cin, H, W = 1, 3, 6, 6
    Cout, KH, KW = 4, 3, 3
    padding = 1
    stride = 1 # Winograd implementation assumes stride=1

    # Check suitability again before creating data
    H_pad_check = H + 2 * padding
    W_pad_check = W + 2 * padding
    if (H_pad_check - 4) % 2 != 0 or (W_pad_check - 4) % 2 != 0:
         print(f"Error: Input size {H}x{W} with padding {padding} gives padded size {H_pad_check}x{W_pad_check}, which is not divisible for F(2x2, 3x3) tiling. Adjust H/W.")
         # Example adjustment logic (might need refinement based on desired output size)
         H = int(np.ceil((H_pad_check - 4) / 2) * 2 + 4 - 2 * padding) if H_pad_check >= 4 else 4 - 2*padding
         W = int(np.ceil((W_pad_check - 4) / 2) * 2 + 4 - 2 * padding) if W_pad_check >= 4 else 4 - 2*padding
         print(f"Adjusted H/W to {H}x{W} for demonstration.")


    x = np.random.randn(N, Cin, H, W).astype(np.float32)
    kernel = np.random.randn(Cout, Cin, KH, KW).astype(np.float32)

    print(f"Input shape: {x.shape}")
    print(f"Kernel shape: {kernel.shape}")

    # Calculate using Winograd
    print("\nCalculating convolution using Winograd F(2x2, 3x3)...")
    try:
        output_winograd = winograd_conv2d_f2x2_3x3(x, kernel, padding=padding, stride=stride)
        print("Winograd output shape:", output_winograd.shape)

        # Calculate using naive convolution for verification
        print("\nCalculating convolution using naive method...")
        output_naive = naive_conv2d(x, kernel, stride_s=stride, padding_p=padding)
        print("Naive output shape:", output_naive.shape)

        # Verify results
        print("\nVerifying results...")
        assert output_winograd.shape == output_naive.shape, "Output shapes mismatch!"
        print("Are Winograd and naive results close?", np.allclose(output_winograd, output_naive, atol=1e-5)) # Use tolerance for float errors
        if not np.allclose(output_winograd, output_naive, atol=1e-5):
            print("Max absolute difference:", np.max(np.abs(output_winograd - output_naive)))

    except ValueError as e:
        print(f"\nError during Winograd calculation: {e}")
    except NotImplementedError as e:
        print(f"\nError: {e}")
