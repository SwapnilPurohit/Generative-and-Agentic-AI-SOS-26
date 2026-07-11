import numpy as np

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices randomly
        np.random.seed(42) # For reproducibility
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: shape (batch_size, num_heads, seq_len, d_k)
        """
        # Q * K^T
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            # Apply mask (e.g., for causal attention)
            scores = np.where(mask == 0, -1e9, scores)
            
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, x, mask=None):
        """
        x: shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Linear projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # 2. Split heads: (batch_size, seq_len, num_heads, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose to get shape (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # 3. Apply Scaled Dot-Product Attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Concatenate heads
        # Transpose back to (batch_size, seq_len, num_heads, d_k) and reshape
        attention_output = attention_output.transpose(0, 2, 1, 3)
        concatenated = attention_output.reshape(batch_size, seq_len, self.d_model)
        
        # 5. Final linear projection
        output = np.matmul(concatenated, self.W_o)
        
        return output, attention_weights

if __name__ == "__main__":
    # Test the Multi-Head Attention block
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    
    # Create random input sequence
    np.random.seed(0)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    output, weights = mha.forward(x)
    
    print("Input shape: ", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    print("\nSample Output (first sequence, first token, first 5 dims):")
    print(output[0, 0, :5])
