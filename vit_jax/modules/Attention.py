import jax
import flax.linen as nn
from typing import Annotated


class Attention(nn.Module):
    dimension: int
    heads: int = 8
    head_dimension: int = 64
    dropout: Annotated[float, "0 <= dropout <= 1.0"] = 0.0

    @nn.compact
    def __call__(self, x: jax.Array, deterministic: False):
        batch_size, sequence_length, _ = x.shape
        scale = self.head_dimension**-0.5
        project_output = not (self.heads == 1 and self.dim_head == self.dim)
        inner_dimension = self.head_dimension * self.heads

        x = nn.LayerNorm()(x)
        qkv = nn.Dense(inner_dimension * 3, use_bias=False)(x)
        q, k, v = jax.numpy.split(qkv, 3, axis=-1)

        def rearrange_to_heads(t):
            return jax.numpy.transpose(
                t.reshape((batch_size, sequence_length, self.heads, self.dim_head)),
                (0, 2, 1, 3),
            )

        q, k, v = map(rearrange_to_heads, (q, k, v))

        dot_products = jax.numpy.matmul(q, jax.numpy.swapaxes(k, -1, -2)) * scale
        attention = jax.nn.softmax(dot_products, axis=-1)
        attention = nn.Dropout(rate=self.dropout, deterministic=deterministic)(
            attention
        )

        output = jax.numpy.matmul(attention, v)
        output = jax.numpy.transpose(output, (0, 2, 1, 3))
        output = output.reshape((batch_size, sequence_length, inner_dimension))

        if project_output:
            output = nn.Dense(self.dim)(output)
            output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(
                output
            )

        return output


if __name__ == "__main__":
    batch_size = 1
    sequence_length = 16
    dimension = 32
    heads = 8

    key = jax.random.PRNGKey(0)
    init_key, data_key, dropout_key = jax.random.split(key, 3)

    dummy_x = jax.random.normal(data_key, (batch_size, sequence_length, dimension))
    attention_module = Attention(dimension=dimension, heads=heads, dropout=0.1)
    variables = attention_module.init(init_key, dummy_x)
    print("✅ Model initialized successfully!")
    print(f"Input shape: {dummy_x.shape}")

    output_eval = attention_module.apply(variables, dummy_x, deterministic=True)
    print(f"Eval output shape: {output_eval.shape}")
    output_train = attention_module.apply(
        variables, dummy_x, deterministic=False, rngs={"dropout": dropout_key}
    )
    print(f"Train output shape: {output_train.shape}")
    assert output_eval.shape == dummy_x.shape, "Output shape should match input shape!"
    print("✅ All tests passed!")
