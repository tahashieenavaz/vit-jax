import jax
from flax.linen import nn
from typing import Annotated

class Attention(nn.Module):
    dimension: int
    heads: int= 8 
    head_dimension: int = 64
    dropout: Annotated[float, "0 <= dropout <= 1.0"] =0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: False):
        batch_size, sequence_size, _ = x.shape
        scale = self.head_dimension ** -0.5
        project_output = not (self.heads == 1 and self.dim_head == self.dim)
        inner_dimension = self.head_dimension * self.heads

        x = nn.LayerNorm()(x)
        qkv = nn.Dense(inner_dimension * 3, use_bias=False)(x)
        q,k,v= jax.numpy.split(qkv, 3, axis=-1)

        def rearrange_to_heads(t):
            return jax.numpy.transpose(t.reshape((batch_size, sequence_size, self.heads, self.dim_head)), (0, 2, 1, 3))
        
        q, k, v = map(rearrange_to_heads, (q, k, v))

        dot_products = jax.numpy.matmul(q, jax.numpy.swapaxes(k, -1, -2)) * scale
        attention = jax.nn.softmax(dot_products, axis=-1)
        attention = nn.Dropout(rate=self.dropout, deterministic=deterministic)(attention)

        output = jax.numpy.matmul(attention, v)
        output = jax.numpy.transpose(output, (0, 2, 1, 3))
        output = output.reshape((batch_size, sequence_size, inner_dimension))

        if project_output:
            output = nn.Dense(self.dim)(output)
            output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(output)

        return output