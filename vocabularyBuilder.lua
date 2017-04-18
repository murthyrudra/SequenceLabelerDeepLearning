
require 'torch'
require 'nn'

function trim(s)
  return s:match'^%s*(.*%S)' or ''
end

function createVocabulary(fileName)
  -- List Of Words --
  local vocabulary = {}
  local reverseVocabulary = {}
  local index  = 1;
  local maxLength = 0;
  
  for line in io.lines(fileName) do
    -- extract every word --
    local currentLength = 0;
    for word in string.gmatch(line,"[^ ]+") do
      currentLength = currentLength +1; 
      if(vocabulary[word] == nil) then
        vocabulary[word] = index;
        
        reverseVocabulary[index] = word
        index = index + 1;
      end
      
      if currentLength > maxLength then
        maxLength = currentLength;
      end
      
    end
    vocabulary["</S>"] = index
    reverseVocabulary[index] = "</S>"
  end
  
  local vocabularySize = 0;
  
  for _ in pairs(vocabulary) do 
    vocabularySize = vocabularySize + 1 
  end
  
  return vocabulary, reverseVocabulary, maxLength, vocabularySize, vocabulary["</S>"]
end

function readCharacterVocabulary(fileName)
  -- List Of Words --
  local characterVocabulary = {}
  local characterReverseVocabulary = {}
  local characterIndex  = 1;
  
--  read a file line by line
  for line in io.lines(fileName) do
--    Don't convert to lowercase, uppercase characters act as features for idenifying NEs
--    line = line:lower()
    local currentLength = 0;
    -- extract every word --
    for word in string.gmatch(line,"[^ \t]+") do
--    need to extract every character
        if word ~= " " then
          if(characterVocabulary[word] == nil) then
            characterVocabulary[word] = characterIndex;
          
            characterReverseVocabulary[word] = word
            characterIndex = characterIndex + 1;
          end
        end
    end  
  end
  
  local vocabularySize = 0;
  
  characterVocabulary["</S>"] = characterIndex
  characterReverseVocabulary[characterIndex] = "</S>"
  
  characterIndex = characterIndex + 1;
  characterVocabulary["<S>"] = characterIndex
  characterReverseVocabulary[characterIndex] = "<S>"
  
  for _ in pairs(characterVocabulary) do
    vocabularySize = vocabularySize + 1 
  end
  
  print("Read "..vocabularySize.." characters ")
  
  return characterVocabulary, characterReverseVocabulary, vocabularySize
end

-- http://www-personal.umich.edu/~rahuljha/files/nlp_from_scratch/ner_embeddings.lua
function loadWordEmbeddings(fileName)
  
  local numberOfLines = 0
  local vocabulary = {}
  local reverseVocabulary = {}
  
  local embeddings = {}
  local dimension = 0
  
  for line in io.lines(fileName) do
    local vector = {}
    local count = 1
    local flag = 0
    
    if numberOfLines == 0 then
      numberOfLines = numberOfLines + 1
    else
      for word in string.gmatch(line,"[^\t ]+") do
        if count == 1 then
          word = word:lower()
          
          if vocabulary[word] ~= nil then
            flag = 1
          else
            vocabulary[word] = numberOfLines
            reverseVocabulary[numberOfLines] = word
          end
        else
          table.insert(vector, tonumber(word))
        end
        count = count + 1
      end
    
      if flag ~= 1 then
        if dimension==0 then
          dimension = #vector
        else
          if dimension ~= #vector then
            print("Error in Dimension "..dimension.." and "..#vector.." "..reverseVocabulary[numberOfLines])
            os.exit()
          end
        end
      
        local embedding = torch.Tensor(vector)
        table.insert(embeddings,embedding)
        numberOfLines = numberOfLines +1
      end
    end
  end
  
  local embedding = torch.zeros(dimension)
  table.insert(embeddings,embedding)
  vocabulary["</S>"] = numberOfLines
  reverseVocabulary[numberOfLines] = "</S>"
 
  embedding = torch.zeros(dimension)
  table.insert(embeddings,embedding)
  vocabulary["<S>"] = numberOfLines
  reverseVocabulary[numberOfLines] = "<S>"

 
  numberOfLines = numberOfLines +1

  local vocabularySize = 0;
  
  for i,v in pairs(vocabulary) do 
    vocabularySize = vocabularySize + 1
--    print(i.."\t"..v) 
  end

  return vocabulary, reverseVocabulary, vocabularySize, embeddings, dimension 
end

function readTagList(fileName)
  -- List Of Words --
  local vocabulary = {}
  local reverseVocabulary = {}
  local index  = 1;
    
  for line in io.lines(fileName) do
    -- extract every word --
    line = line:lower()
    if(vocabulary[line] == nil) then
      vocabulary[line] = index;
      
      reverseVocabulary[index] = line
      index = index + 1;
    end
  end
  
  local vocabularySize = 0;
  
  for i,v in pairs(vocabulary) do 
    vocabularySize = vocabularySize + 1
    print(i.."\t"..v) 
  end
  
  print("Read "..vocabularySize.." number of tags")
    
  return vocabulary, reverseVocabulary, vocabularySize
end


function readGazList(fileName)
  -- List Of Words --
  local vocabulary = {}
  local reverseVocabulary = {}
  local index  = 1;
    
  for line in io.lines(fileName) do
    -- extract every word --
    line = line:lower()
    if(vocabulary[line] == nil) then
      vocabulary[line] = index;
      
      reverseVocabulary[index] = line
      index = index + 1;
    end
  end
  
  local vocabularySize = 0;
  for i,v in pairs(vocabulary) do
    vocabularySize = vocabularySize + 1
  end
  
  print("Read "..vocabularySize.." number of gazetteers")
    
  return vocabulary, reverseVocabulary, vocabularySize
end
