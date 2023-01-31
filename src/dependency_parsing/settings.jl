export Settings;

struct Settings
  embeddings_size::Integer
  hidden_size::Integer
  batch_size::Integer
  reg_weight::Float64
  sample_size::Integer

  Settings(;
    embeddings_size::Integer=100,
    hidden_size::Integer=300,
    batch_size::Integer=32,
    reg_weight=1e-7,
    sample_size=10000,
  ) = new(embeddings_size, hidden_size, batch_size, reg_weight, sample_size)
end