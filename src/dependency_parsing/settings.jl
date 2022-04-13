export Settings;

struct Settings
  embeddings_size::Integer
  hidden_size::Integer
  batch_size::Integer

  Settings(;
    embeddings_size::Integer=300,
    hidden_size::Integer=200,
    batch_size::Integer=48
  ) = new(embeddings_size, hidden_size, batch_size)
end