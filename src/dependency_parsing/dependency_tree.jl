const EMPTY_LABEL = "EMPTY_LABEL"
const EMPTY_NODE = -1

mutable struct TreeNode
  word_id::Integer
  label::String
  head_id::Integer

  TreeNode(word_id::Integer) = new(word_id, EMPTY_LABEL, EMPTY_NODE)
end

# Root node will have id == 0, others nodes id depend from their position in sentence
mutable struct DependencyTree
  root::TreeNode
  nodes::Vector{TreeNode}

  function DependencyTree(sentence::Vector{Tuple{String, String}})
    nodes = map(enumerate(sentence)) do (index, _)
      TreeNode(index)
    end

    root = TreeNode(0)

    new(root, nodes)
  end
end

function set_arc(tree::DependencyTree, word_id::Integer, head_id::Integer, label::String)
  if word_id < 1 || word_id > length(tree.nodes)
    return
  end

  if head_id < 0 || head_id > length(tree.nodes)
    return
  end

  dependent_node = tree.nodes[word_id]
  dependent_node.head_id = head_id
  dependent_node.label = label
end

function has_head(tree::DependencyTree, word_id::Integer)
  if word_id == 0
    return false
  end

  node = tree.nodes[word_id]

  node.head_id != EMPTY_NODE
end