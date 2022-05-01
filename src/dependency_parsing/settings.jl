export Settings;

struct Settings
  embeddings_size::Integer
  hidden_size::Integer
  batch_size::Integer
  reg_weight::Float64
  sample_size::Integer

  Settings(;
    embeddings_size::Integer=300,
    hidden_size::Integer=200,
    batch_size::Integer=48,
    reg_weight=1e-8,
    sample_size=5000
  ) = new(embeddings_size, hidden_size, batch_size, reg_weight, sample_size)
end