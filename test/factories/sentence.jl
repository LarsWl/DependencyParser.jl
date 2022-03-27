using DependencyParser.Core

# Sentence: He wrote her a letter.
function build_sentence()
  words = ["He", "wrote", "her", "a", "letter", "."]

  tokens = map(id -> Token(id, words[id]), collect(1:length(words)))
  tags = map(tag -> PosTag(tag), ["PRP", "VBN", "PRP", "DT", "NN", "JJ"])
  
  Sentence(tokens, tags, length(words))
end
