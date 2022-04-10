@testset "Connlu utils" begin
  
  filename = "test/fixtures/connlu_test.txt"
  correct_sentences = [
    "Attribution-ShareAlike 2.0.",
    "Any use of the work other than as authorized under this license or copyright law is prohibited.",
    "UNLESS OTHERWISE AGREED TO BY THE PARTIES IN WRITING, LICENSOR OFFERS THE WORK AS-IS AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE MATERIALS, EXPRESS, IMPLIED, STATUTORY OR OTHERWISE, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTIBILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OF ABSENCE OF ERRORS, WHETHER OR not DISCOVERABLE.",
  ]
  load_result = DependencyParser.DependencyParsing.load_connlu_file(filename)
  
  foreach(correct_sentences) do correct_sentence
    @test correct_sentence in map(connlu_s -> connlu_s.string_doc.text, load_result)
  end
end