export TreeNode;
export EMPTY_LABEL, EMPTY_NODE

const EMPTY_LABEL = "EMPTY_LABEL"
const EMPTY_NODE = -1

mutable struct TreeNode
  word_id::Integer
  label::String
  head_id::Integer

  TreeNode(word_id::Integer) = new(word_id, EMPTY_LABEL, EMPTY_NODE)
  TreeNode(word_id::Integer, label::String, head_id::Integer) = new(word_id, label, head_id)
end