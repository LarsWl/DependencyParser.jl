export Settings;

struct Settings
  embedding_size::Integer
  hidden_size::Integer
  batch_size::Integer

  Settings() = new(300, 200, 48)
end