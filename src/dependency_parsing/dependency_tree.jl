export DependencyTree
export set_arc, del_arc, has_head, arc_present

# Root node will have id == 0, others nodes id depend from their position in sentence
mutable struct DependencyTree
  root::TreeNode
  nodes::Vector{TreeNode}
  length::Integer

  DependencyTree(root::TreeNode, nodes::Vector{TreeNode}, length::Integer) = new(root, nodes, length)
  function DependencyTree(sentence::Sentence)
    nodes = map(collect(1:sentence.length)) do index
      TreeNode(index, sentence.tokens[index].name)
    end

    root = TreeNode(0)

    new(root, nodes, length(nodes))
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

function del_arc(tree::DependencyTree, word_id::Integer)
  if word_id < 1 || word_id > length(tree.nodes)
    return
  end

  dependent_node = tree.nodes[word_id]
  dependent_node.head_id = EMPTY_NODE
  dependent_node.label = EMPTY_LABEL
end

function has_head(tree::DependencyTree, word_id::Integer)
  if word_id == 0
    return false
  end

  node = tree.nodes[word_id]

  node.head_id != EMPTY_NODE
end

function arc_present(tree::DependencyTree, head_id::Integer, word_id::Integer)
  if 1 <= word_id <= tree.length
    return tree.nodes[word_id].head_id == head_id
  end
  
  false
end

function get_left_child(tree::DependencyTree, word_id::Integer; child_number = 1)
   (word_id < 0 || word_id > tree.length) && return NONEXIST_TOKEN

   num = 0

   for child_id = 1:(word_id - 1)
    tree.nodes[child_id].head_id != word_id && continue

    num += 1
    num == child_number && return child_id
   end

   NONEXIST_TOKEN
end

function get_right_child(tree::DependencyTree, word_id::Integer; child_number = 1)
  (word_id < 0 || word_id > tree.length) && return NONEXIST_TOKEN

  num = 0

  for child_id = (word_id + 1):tree.length
   tree.nodes[child_id].head_id != word_id && continue

   num += 1
   num == child_number && return child_id
  end

  NONEXIST_TOKEN
end

function convert_to_string(tree::DependencyTree)
  lines = map(tree.nodes) do node
    head_word = if node.head_id == 0
      tree.root.token
    elseif node.head_id > 0
      tree.nodes[node.head_id].token
    else
      NO_HEAD
    end

    "$(node.label)($(head_word)-$(node.head_id), $(node.token)-$(node.word_id))"
  end

  join(lines, "\n")
end