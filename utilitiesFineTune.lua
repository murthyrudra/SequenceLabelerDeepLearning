function split(str, pat)
   local t = {}  -- NOTE: use {n = 0} in Lua-5.0
   local fpat = "(.-)" .. pat
   local last_end = 1
   local s, e, cap = str:find(fpat, 1)
   while s do
      if s ~= 1 or cap ~= "" then
   table.insert(t,cap)
      end
      last_end = e+1
      s, e, cap = str:find(fpat, last_end)
   end
   if last_end <= #str then
      cap = str:sub(last_end)
      table.insert(t, cap)
   end
   return t
end

-- http://lua-users.org/wiki/FileInputOutput

-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

function trim(s)
  return s:match'^%s*(.*%S)' or ''
end

--http://stackoverflow.com/questions/19664666/check-if-a-string-isnt-nil-or-empty-in-lua
function isEmpty(s)
  return s == nil or s == ''
end

-- http://stackoverflow.com/questions/5178087/how-do-i-get-the-highest-integer-in-a-table-in-lua
function max(t)
    if #t == 0 then return nil, nil end
    local key, value = 1, t[1]
    for i = 2, #t do
        if value > t[i] then
            key, value = i, t[i]
        end
    end
    return key, value
end

function loadDataNew(fileName,sourceDictionary, tagVocabulary,reverseSourceDictionary, reverseTagVocabulary, delimiter, wordColumnId, nerTagColumnId, characterVocabulary,
  characterVocabularySize, maxCharacternGramSequence , minCharacternGramSequence, charTokenField)
    
  local sourceData = {}
  local sourceTags = {}
  local sourceWordPresent = {}
  local characterData = {}
  local characterAsItIs = {}
  
  local sourceSentence = {}
  
  local wordsNotFound = 0;
  local totalWords = 0;

  local lineWords = {}
  local lineTags = {}
  local lineWordPresent = {}
  local lineCharacters = {}
  
  local dataAsItIs = {}
  local lineAsItIs = {}
  
  local maxSequenceLength = 0
  
  for line in io.lines(fileName) do
--    remove leading and trailing whitespaces
    line = trim(line)
    
--    Check if it's a blank line or not
    if isEmpty(line) then
--    Either the list is empty or list contains a line
      if #lineWords ~= 0 then
        if #lineWords ~= #lineTags then
          print("Error mismatch words and tags ")
          print(lineWords)
          print(lineTags)
          os.exit()
        end
        
        local itemp = {}
        
        for i = 1,#lineWords do
          itemp[i] = lineWords[i]
        end
        table.insert(sourceData,itemp)
        
        itemp = torch.Tensor(#lineTags)
        for i = 1,#lineTags do
          itemp[i] = lineTags[i]
        end
        table.insert(sourceTags,itemp)
        
        if #lineWords > maxSequenceLength then
          maxSequenceLength = #lineWords
        end
        
        itemp = {}
        
--      for every word in the line
--        for i=1,#lineCharacters do
--          itemp[i] = {}
--          for j =minCharacternGramSequence,maxCharacternGramSequence do
--            if #lineCharacters[i] >= j then
--              itemp[i][j-minCharacternGramSequence+1] = torch.zeros(#lineCharacters[i], characterVocabularySize)
              
--              for k=1,#lineCharacters[i] do
--                if lineCharacters[i][k] ~= 0 then
--                  itemp[i][j-minCharacternGramSequence+1][k][lineCharacters[i][k]] = 1.0
--                end
--              end 
--            else
--              itemp[i][j-minCharacternGramSequence+1] = torch.zeros(j, characterVocabularySize)
              
--              for k=1,#lineCharacters[i] do
--                if lineCharacters[i][k] ~= 0 then
--                  itemp[i][j-minCharacternGramSequence+1][k][lineCharacters[i][k]] = 1.0
--                end
--              end
              
--              for k=#lineCharacters[i],j do
--                itemp[i][j-minCharacternGramSequence+1][k][characterVocabulary["</S>"]] = 1.0
--              end
--            end
--          end
--          itemp[i] = lineCharacters[i]
--        end
        
        
        table.insert(characterData,lineCharacters)
        
        table.insert(dataAsItIs,lineAsItIs)
        
        table.insert(sourceWordPresent,lineWordPresent)
      end
      
      lineWordPresent = {}
      lineAsItIs = {}
      lineWords = {}
      lineTags = {}
      lineCharacters = {}
      sourceSentence = {}
      
    else
      -- extract every token and add it to the list--
      
      local sourceWord = {}
      local index = 0
      for word in string.gmatch(line,"[^\t]+") do
        table.insert(sourceWord, word)
      end
      
--      convert words to indices and add to the table
      local input = torch.IntTensor(1)
      if sourceDictionary[sourceWord[wordColumnId]:lower()] ~= nil then
        table.insert(lineWordPresent,1)
        input[1] = sourceDictionary[sourceWord[wordColumnId]:lower()]
      else
        wordsNotFound = wordsNotFound + 1
        table.insert(lineWordPresent,0)
        input[1] = sourceDictionary["</S>"]
      end
      totalWords = totalWords + 1
      
      table.insert(lineWords, input)

--      convert tags to indices and add to the table      
      if tagVocabulary[sourceWord[nerTagColumnId]:lower()] ~= nil then
        table.insert(lineTags, tagVocabulary[sourceWord[nerTagColumnId]:lower()])
      else
        print("Error tag "..sourceWord[nerTagColumnId]:lower().." not found")
        print(line)
        os.exit()
      end
      
      local lineForOutput = sourceWord[wordColumnId].."\t"..sourceWord[nerTagColumnId]
      table.insert(lineAsItIs, lineForOutput)
      
--      read character one by one and add it to wordCharacter
      local wordCharacter = {}
      table.insert(wordCharacter, characterVocabulary["<S>"])

      for character in string.gmatch(sourceWord[charTokenField],"[^ ]+") do
        if(characterVocabulary[character] ~= nil) then
            table.insert(wordCharacter,characterVocabulary[character])
          else
--            print("Error character "..character.." not found")
--            table.insert(wordCharacter,0)
--            os.exit()
          end

      end
      table.insert(wordCharacter, characterVocabulary["</S>"])
            
      table.insert(lineCharacters,wordCharacter)
    end
    
  end
    
  print(totalWords.."\t"..wordsNotFound)
  
  return sourceData, sourceTags, dataAsItIs, maxSequenceLength, sourceWordPresent, characterData 
end

function loadDataTest(fileName,sourceDictionary, tagVocabulary,reverseSourceDictionary, reverseTagVocabulary, delimiter, characterVocabulary,
  characterVocabularySize, maxCharacternGramSequence , minCharacternGramSequence)
    
  local wordColumnId = 1
  local characterColumnId = 2
  local nerColumnId = 3
  
  local sourceData = {}
  local sourceWordPresent = {}
  local characterData = {}
  local characterAsItIs = {}
  
  local sourceSentence = {}
  
  local wordsNotFound = 0;
  local totalWords = 0;

  local lineWords = {}
  local lineWordPresent = {}
  local lineCharacters = {}
  
  local dataAsItIs = {}
  local lineAsItIs = {}
  
  local maxSequenceLength = 0
  
  for line in io.lines(fileName) do
--    remove leading and trailing whitespaces
    line = trim(line)
    
--    Check if it's a blank line or not
    if isEmpty(line) then
--    Either the list is empty or list contains a line
      if #lineWords ~= 0 then
        if #lineWords ~= #lineCharacters then
          print("Error mismatch words and tags ")
          print(lineWords)
          print(lineCharacters)
          os.exit()
        end
        
        local itemp = {}
        
        for i = 1,#lineWords do
          itemp[i] = lineWords[i]
        end
        table.insert(sourceData,itemp)
        
        if #lineWords > maxSequenceLength then
          maxSequenceLength = #lineWords
        end
        
        itemp = {}
        
        table.insert(characterData,lineCharacters)
        
        table.insert(dataAsItIs,lineAsItIs)
        
      end
      
      lineAsItIs = {}
      lineWords = {}
      lineCharacters = {}
      
    else
      -- extract every token and add it to the list--
      
      local sourceWord = {}
      local index = 0
      for word in string.gmatch(line,"[^\t]+") do
        table.insert(sourceWord, word)
      end
      
--      convert words to indices and add to the table
      local input = torch.IntTensor(1)
      if sourceDictionary[sourceWord[wordColumnId]:lower()] ~= nil then
        input[1] = sourceDictionary[sourceWord[wordColumnId]:lower()]
      else
        wordsNotFound = wordsNotFound + 1
        input[1] = sourceDictionary["</S>"]
      end
      totalWords = totalWords + 1
      
      table.insert(lineWords, input)

      local lineForOutput = sourceWord[wordColumnId].."\t"..sourceWord[characterColumnId]
      table.insert(lineAsItIs, lineForOutput)
      
--      read character one by one and add it to wordCharacter
      local wordCharacter = {}
      table.insert(wordCharacter, characterVocabulary["<S>"])

      for character in string.gmatch(sourceWord[characterColumnId],"[^ ]+") do
        if(characterVocabulary[character] ~= nil) then
          table.insert(wordCharacter,characterVocabulary[character])
        end

      end
      table.insert(wordCharacter, characterVocabulary["</S>"])
      
      
      table.insert(lineCharacters,wordCharacter)
    end
    
  end
    
  print(totalWords.."\t"..wordsNotFound)
  
  return sourceData, dataAsItIs, maxSequenceLength, characterData 
end