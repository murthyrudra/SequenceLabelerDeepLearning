---- Deep Learning Module for Sequence Labeling tasks like NER, POS Tagging
-- A LSTM layer is enclosed in a BiSequencer to capture both left and right contexts
-- Additionally character sequence is sent through a CNN layer to learn character-level features
-- Here, we learn to compose nGram characters
-- This is enclosed by a dropout layer for regularization
-- Previous tag + output from BiSequencer is sent through a softmaxlayer for prediction

require 'torch'
require 'nn'
require 'sys'
require 'optim'
require 'xlua'   
require 'rnn'
require 'pprint'
local nninit = require 'nninit'
require 'vocabularyBuilder'
require 'utilitiesFineTune'
require 'beamSearch'
require 'LinearNB'
manifol = require 'manifold'
torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a RNN Sequence Labeller')
cmd:text()
cmd:text('Options')
cmd:option('-train',"","Path to Train File")
cmd:option('-tune',"","Path to Development File")
cmd:option('-test',"","Path to Test File")
cmd:option('-embeddings',"","Path to Embeddings File")
cmd:option('-characterSource',"","Used to Build Character Vocabulary")
cmd:option('-tags',"","Tag List")
cmd:option('-useGPU',false,"Use GPU")
cmd:option('-hiddenSize',200,"Size of Hidden Layer Neurons")
cmd:option('-predict',false,"Perform Only Prediction, no training")
cmd:option('-language',"","Language ")
cmd:option('-modelInfo',"","Model Info used as folder name ")
cmd:option('-preLoadModule',5,"intermediate module epoch number")
cmd:option('-intermediate',false,"Start training using pretrained module")
cmd:option('-optimizationTechnique',0, "Optimization Technique to be used")
cmd:option('-delimiter', " ", "Delimiter between fields in CoNLL Format, default is space")
cmd:option('-wordTokenField', 1, "Column Number of word in CoNLL Format")
cmd:option('-charTokenField',2, "")
cmd:option('-nerTagField', 3, "Column Number of NER Tags in CoNLL Format")
cmd:option('-characterDimension', 50, "Number of features extracted from characters")
cmd:option('-maxCharacternGrams', 5, "Number of maximum character n-grams to look at")
cmd:option('-minCharacternGrams', 2, "Number of minimum character n-grams to look at")
cmd:option('-tagTestSet', false, "Perform tagging on test set i.e, the test set contains no tag information")
cmd:option('-useOnline', false, "Perform Online Prediction")
cmd:option('-lr',0.1, "Initial Learning Rate")
cmd:option('-unsupervised',false,"unsupervised train model")
cmd:option('-unsupPath',"", "path to unsupervised train model")
cmd:option('-unsupEpoch',2,"epoch number of unsupervised train model")
cmd:option('-printGraph',false,"")
cmd:text()

local options = cmd:parse(arg)


if not options.tagTestSet then
  print("using Full Source File "..options.embeddings)
  print("using Train File "..options.train)
  print("using Tune File "..options.tune)
  print("using Test File "..options.test)
  print("Using Character Source File to create character vocabulary "..options.characterSource)
  print("using Hidden Layer Neurons "..options.hiddenSize)
  print("Training for language "..options.language)

  print("Using token delimiter in CoNLL Format "..options.delimiter)
  print("Column Number of Words in CoNLL Format "..options.wordTokenField)
  print("Column Number of nerTags in CoNLL Format "..options.nerTagField)
  print("Number of features extracted from characters "..options.characterDimension)
end


torch.manualSeed(1)
if options.useGPU then
  local ok, cunn = pcall(require, 'cunn')
  local ok2, cutorch = pcall(require, 'cutorch')
  if not ok then print('package cunn not found!') end
  if not ok2 then print('package cutorch not found!') end
  if ok and ok2 then
      print('using CUDA on GPU ')
  else
      options.useGPU = false
      print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
      print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
      print('Falling back on CPU mode')
  end
else
  print("using CPU ")
end

--Read Source Vocabulary and word embeddings--
local sourceDictionary, reverseSourceDictionary, sourceDictionarySize, embeddings, embeddingDimension = loadWordEmbeddings(options.embeddings)
print("Read words with word embedding dimension "..embeddingDimension.." and number of entries "..sourceDictionarySize)

--Read Tag List
local tagVocabulary, tagReverseVocabulary, numberOfTags = readTagList(options.tags)

--Read character vocabulary
local characterVocabulary, reverseCharacterVocabulary, characterVocabularySize = readCharacterVocabulary(options.characterSource) 

local maxCharacternGramsequence = options.maxCharacternGrams
local minCharacternGramsequence = options.minCharacternGrams

local trainDataSource, trainDataTags, trainDataAsItIs, trainMaxSequenceLength, trainWordPresent, trainDataCharacter, traincharacterAsItIs

local tuneDataSource, tuneDataTags, tuneDataAsItIs, tuneMaxSequenceLength, tuneWordPresent, tuneDataCharacter, tunecharacterAsItIs

local testDataSource, testDataTags, testDataAsItIs, testMaxSequenceLength, testWordPresent, testDataCharacter, testcharacterAsItIs

if options.useOnline then
  
else
  if options.tagTestSet then
  testDataSource, testDataAsItIs, testMaxSequenceLength, testDataCharacter= 
        loadDataTest(options.test, sourceDictionary, tagVocabulary, reverseSourceDictionary, tagReverseVocabulary, options.delimiter,       characterVocabulary, characterVocabularySize
        , maxCharacternGramsequence, minCharacternGramsequence)
      
  else
    if not options.predict then
      trainDataSource, trainDataTags, trainDataAsItIs, trainMaxSequenceLength, trainWordPresent, trainDataCharacter = 
          loadDataNew(options.train, sourceDictionary, tagVocabulary, reverseSourceDictionary, tagReverseVocabulary, options.delimiter, options.wordTokenField, options.nerTagField, 
          characterVocabulary, characterVocabularySize, maxCharacternGramsequence, minCharacternGramsequence, options.charTokenField)
      print("Training data contains "..#trainDataSource.." Lines ".." maximum number of words in a line is "..trainMaxSequenceLength)

        tuneDataSource, tuneDataTags, tuneDataAsItIs, tuneMaxSequenceLength, tuneWordPresent, tuneDataCharacter =
          loadDataNew(options.tune, sourceDictionary, tagVocabulary, reverseSourceDictionary, tagReverseVocabulary, options.delimiter, options.wordTokenField, options.nerTagField,
           characterVocabulary, characterVocabularySize, maxCharacternGramsequence, minCharacternGramsequence, options.charTokenField)
      print("Development data contains "..#tuneDataSource.." Lines")

    end
    
  testDataSource, testDataTags, testDataAsItIs, testMaxSequenceLength, testWordPresent, testDataCharacter, testcharacterAsItIs = 
        loadDataNew(options.test, sourceDictionary, tagVocabulary, reverseSourceDictionary, tagReverseVocabulary, options.delimiter, options.wordTokenField, options.nerTagField,
         characterVocabulary, characterVocabularySize, maxCharacternGramsequence, minCharacternGramsequence, options.charTokenField)
  end
  print("Test data contains "..#testDataSource.." Lines")
end


--read word embeddings and load into lookup table
local embed = nn.LookupTable(sourceDictionarySize, embeddingDimension, 0, 1.0, 2) 
for i=1,#embeddings do
  embed.weight[i] = embeddings[i]:clone()
end

--for i=1,#embeddings do
--  embeddings[i] = {}
--end
--embed:init('weight', nninit.xavier,{dist='normal'})

local wordRep = nn.Sequential():add(embed):add(nn.Squeeze())
print("Finished initializing lookup Table")

---- Print Word Embeddings for Debugging purpose
--for i,v in pairs(sourceDictionary) do
--  print(i.."\t"..v)
--  local input = torch.Tensor(1,1)
--  input[1][1] = sourceDictionary[i]
--  local e = embed:forward(input[1])
--  print(e)
--end

-- Extract Character Level Features: Look at single character
local charLearner = {}
local ngramFeatureLearner = {}
for i = minCharacternGramsequence,maxCharacternGramsequence do
  ngramFeatureLearner[i-minCharacternGramsequence +1] = nn.TemporalConvolution(characterVocabularySize, options.characterDimension, i)
  ngramFeatureLearner[i-minCharacternGramsequence +1]:init('weight', nninit.xavier, {dist = 'normal'})
  
  charLearner[i-minCharacternGramsequence +1] = nn.Sequential()
  charLearner[i-minCharacternGramsequence +1]:add(ngramFeatureLearner[i-minCharacternGramsequence +1])
  charLearner[i-minCharacternGramsequence +1]:add(nn.Max(1))
end


local extractFeatures = nn.ParallelTable()
extractFeatures:add(wordRep)
for i =minCharacternGramsequence,maxCharacternGramsequence do
  extractFeatures:add(charLearner[i-minCharacternGramsequence +1])
end

local concatFeatures = nn.Sequential()
concatFeatures:add(extractFeatures)
concatFeatures:add(nn.JoinTable(1))

local memory = nn.Sequential()
memory:add(nn.Sequencer(concatFeatures))
memory:add(nn.Sequencer(nn.Dropout()))

local lstm = nn.FastLSTM(embeddingDimension + (maxCharacternGramsequence - minCharacternGramsequence +1) * options.characterDimension,options.hiddenSize)
local lstm1 = nn.FastLSTM(embeddingDimension + (maxCharacternGramsequence - minCharacternGramsequence +1) * options.characterDimension,options.hiddenSize)

local memory1 = nn.Sequential()
memory1:add(nn.BiSequencer(lstm, lstm1))


--Sequencer for Top Layer, no longer needed makes testing difficult
local sequenceLabeler = nn.Sequential()
----This architecture gives the best results as of now
sequenceLabeler:add(nn.LinearNB(options.hiddenSize * 2 + numberOfTags, numberOfTags))
sequenceLabeler:add(nn.LogSoftMax())

--Error function for NER
local criterion = nn.ClassNLLCriterion()
criterion.sizeAverage = false

if options.useGPU then
  memory:cuda()
  memory1:cuda()
  sequenceLabeler:cuda()
  criterion:cuda()
end

-- put the above things into one flattened parameters tensor
local parameters, grad_params
parameters, grad_params = nn.Container():add(sequenceLabeler):add(memory):add(memory1):getParameters()

-- Use SGD with Nesterov Momentum
local optimState = {
      learningRate = options.lr,
      weightDecay = 0.0,
      momentum = 0.001,
      learningRateDecay = 0.0,
      nesterov = true,
      dampening = 0.0,
      alpha = 0.95
   }
local optimMethod
if options.optimizationTechnique == 0 then
   optimMethod = optim.sgd
   print("Using SGD with Momentum")
elseif options.optimizationTechnique == 1 then
  optimMethod = optim.adagrad
  print("Using Adagrad")
elseif options.optimizationTechnique == 2 then
  optimMethod = optim.rmsprop
  print("Using RMSprop")
end

function performTraining(inputSource, target, characterInput)
------------------- forward pass -------------------
  local loss = 0
  
  if #inputSource ~= target:size(1) then
    print("Size mismatch "..#inputSource.."\t"..target:size(1))
    os.exit()
  end

--Start from scratch
  memory:forget()
  memory1:forget()
  
 --Combine embeddings and character input  
  local inputToMemory = {}
  for t=1,#inputSource do
    inputToMemory[t] = {}
    if options.useGPU then
      inputToMemory[t][1] = inputSource[t]:cuda()
      
      for i=minCharacternGramsequence,maxCharacternGramsequence do
        inputToMemory[t][i -minCharacternGramsequence+2] = characterInput[t][i -minCharacternGramsequence+1]:clone():cuda()
      end
      
    else
      inputToMemory[t][1] = inputSource[t]:clone()
      for i=minCharacternGramsequence,maxCharacternGramsequence do
        inputToMemory[t][i -minCharacternGramsequence+2] = characterInput[t][i -minCharacternGramsequence+1]:clone()
      end
    end
  end
  
--Send embeddings and character input to obtain sequence of representation
  local memoryActivations = memory:forward(inputToMemory)
  
  local memoryActivations1 = memory1:forward(memoryActivations)
  
--This is fed to softmax layer for prediction
  local inputVector = torch.zeros(#inputSource, options.hiddenSize * 2 + numberOfTags)
  for t=1,#inputSource do
    local previousOutput = torch.zeros(numberOfTags)
--Send Previous tag as input
    if t~= 1 then
      previousOutput[target[t-1]] = 1.0
    end
    inputVector[t] = torch.cat(memoryActivations1[t]:float():clone(), previousOutput:clone())
  end
  
  if options.useGPU then
     inputVector = inputVector:cuda()
  end
    
--Obtain predictions
  local predictions = sequenceLabeler:forward(inputVector)
  
--Calculate loss
  if options.useGPU then
    loss = loss + criterion:forward(predictions, target:cuda())
  else
    loss = loss + criterion:forward(predictions, target)
  end
  
----Backward Pass
  
--Calculate error w.r.t. criterion
  local gradOutputs
  if options.useGPU then
    gradOutputs = criterion:backward(predictions, target:cuda())
  else
    gradOutputs = criterion:backward(predictions, target)
  end

--Backpropagate this error to softmax layer
  local gradTopLayer = sequenceLabeler:backward(inputVector, gradOutputs)

--Calculate gradient w.r.t. sequencer layer
  local gradMemory = {}
  for t=1,#inputSource do
    if options.useGPU then
      gradMemory[t] = gradTopLayer[t]:float():narrow(1, 1, options.hiddenSize * 2):cuda()
    else
      gradMemory[t] = gradTopLayer[t]:float():narrow(1, 1, options.hiddenSize * 2)
    end
  end

--Backpropagate this error to sequencer layer
  local gradInput = memory1:backward(memoryActivations, gradMemory)
  memory:backward(inputToMemory, gradInput)
  
  loss = loss / #inputSource
  return loss
end

function extractCharacterInput(lineCharacters)
  itemp = {}
  
  for i=1,#lineCharacters do
    itemp[i] = {}
    for j =minCharacternGramsequence,maxCharacternGramsequence do
      if #lineCharacters[i] >= j then
        itemp[i][j-minCharacternGramsequence+1] = torch.zeros(#lineCharacters[i], characterVocabularySize)
        
        for k=1,#lineCharacters[i] do
          if lineCharacters[i][k] ~= 0 then
            itemp[i][j-minCharacternGramsequence+1][k][lineCharacters[i][k]] = 1.0
          end
        end 
      else
        itemp[i][j-minCharacternGramsequence+1] = torch.zeros(j, characterVocabularySize)
        
        for k=1,#lineCharacters[i] do
          if lineCharacters[i][k] ~= 0 then
            itemp[i][j-minCharacternGramsequence+1][k][lineCharacters[i][k]] = 1.0
          end
        end
        
        for k=#lineCharacters[i],j do
          itemp[i][j-minCharacternGramsequence+1][k][characterVocabulary["</S>"]] = 1.0
        end
      end
    end
  end

  return itemp
end

function validation(DataSource, DataTags, CharacterDataSource)
  local lineCount = 0;
  local loss = 0
    
  for eachSentence =1,#DataSource do
--Each sentence is a new sequence and the state needs to be reset
    memory:evaluate()
    memory:forget()
    
    memory1:evaluate()
    memory1:forget()
    
    local input = DataSource[eachSentence]
    local target = DataTags[eachSentence]
    local characterInput = extractCharacterInput(CharacterDataSource[eachSentence])
    
    grad_params:zero()
    
--Combine embeddings and character sequence input
    local inputToMemory = {}
    for t=1, #input do
      inputToMemory[t] = {}
      if options.useGPU then
        inputToMemory[t][1] = input[t]:cuda()
      
        for i=minCharacternGramsequence,maxCharacternGramsequence do
          inputToMemory[t][i - minCharacternGramsequence+2] = characterInput[t][i -minCharacternGramsequence +1]:clone():cuda()
        end
      
      else
        inputToMemory[t][1] = input[t]:clone()
        for i=minCharacternGramsequence,maxCharacternGramsequence do
          inputToMemory[t][i - minCharacternGramsequence+2] = characterInput[t][i -minCharacternGramsequence +1]:clone()
        end
      end
    end
    --pprint(inputToMemory)
--send the combined input through LSTM layer to obtain representations
    local memoryActivations = memory:forward(inputToMemory)
    local memoryActivations1 = memory1:forward(memoryActivations)

--the representations are combined with previous tag information for prediction
    local inputVector = torch.zeros(#input, options.hiddenSize * 2 + numberOfTags)
    for t=1,#input do
      local previousOutput = torch.zeros(numberOfTags)
      if t~= 1 then
        previousOutput[target[t-1]] = 1.0
      end
      inputVector[t] = torch.cat(memoryActivations1[t]:float():clone(), previousOutput:clone())
    end
    
    if options.useGPU then
       inputVector = inputVector:cuda()
    end
    
    local predictions = sequenceLabeler:forward(inputVector)
    
    if options.useGPU then
      loss = loss + criterion:forward(predictions, target:cuda())
    else
      loss = loss + criterion:forward(predictions, target)
    end
  end
  
  print("Cost on Development set "..(loss/#DataSource))
  return (loss/#DataSource)
  
end

function test(DataSource, DataTags, DataAsItIs, epoch, wordPresent, CharacterDataSource, CharacterAsItIs)
--  local fN = assert(io.open("output/"..options.language.."/"..options.modelInfo.."/embedding", "w"))
--
--   for i=1,sourceDictionarySize do
--     fN:write(reverseSourceDictionary[i].." ")
--    
--     local e = embed.weight[i]:clone():float()
--    
--     for j=1,e:size(1) do
--      fN:write(e[j].." ")
--     end
--     fN:write("\n")
--   end  
  
  local f = assert(io.open("output/"..options.language.."/"..options.modelInfo.."/"..options.language..".out__"..epoch, "w"))
  print("output/"..options.language.."/"..options.modelInfo.."/"..options.language.."out__"..epoch)
  
  local loss = 0
  
  local cm = optim.ConfusionMatrix(tagReverseVocabulary)
  cm:zero()
  
  local cmKnown = optim.ConfusionMatrix(tagReverseVocabulary)
  cmKnown:zero()
  
  local cmUnknown = optim.ConfusionMatrix(tagReverseVocabulary)
  cmUnknown:zero()
  
  local presentAndCorrect = 0
  local presentAndWrong = 0
  
  local absentAndCorrect = 0
  local absentAndWrong = 0
  
  for pairIterate =1,#DataSource do
    -- extract every word --
    
    if pairIterate%1000 == 0 then
      collectgarbage()
    end
    
    xlua.progress(pairIterate, #DataSource)

--Each sentence is a new sequence and the state needs to be reset    
    memory:evaluate()
    memory:forget()
    
    memory1:evaluate()
    memory1:forget()
    
    local input = DataSource[pairIterate]
    local linePresent
    local target
    local characterInput 
    if not options.tagTestSet then
      target = DataTags[pairIterate]
      linePresent = wordPresent[pairIterate]
      characterInput = extractCharacterInput(CharacterDataSource[pairIterate])
    else
      characterInput = extractCharacterInput(CharacterDataSource[pairIterate])
    end
    
    grad_params:zero()
    
    local predictions = {}
    
-- Send the input sequence through a Lookup Table to obtain it's embeddings
    
--Combine embeddings and character sequence input
    local inputToMemory = {}
    for t=1,#input do
      inputToMemory[t] = {}
      
      if options.useGPU then
        inputToMemory[t][1] = input[t]:cuda()
      
        for i=minCharacternGramsequence,maxCharacternGramsequence do
          inputToMemory[t][i - minCharacternGramsequence+2] = characterInput[t][i -minCharacternGramsequence +1]:clone():cuda()
        end
      
      else
        inputToMemory[t][1] = input[t]:clone()
        for i=minCharacternGramsequence,maxCharacternGramsequence do
          inputToMemory[t][i - minCharacternGramsequence+2] = characterInput[t][i -minCharacternGramsequence +1]:clone()
        end
      end
    end
    
--send the combined input through LSTM layer to obtain representations
    local memoryActivations = memory:forward(inputToMemory)
    local memoryActivations1 = memory1:forward(memoryActivations)
    
--Do the actual sequence tagging using beamsearch
    local predictedSequence,cost = doBeamSearch(options, memoryActivations1, numberOfTags, options.useGPU, #input, sequenceLabeler)
    
    local predictedOutput = {}
    local isKnownWordError = false
    for i=1, #predictedSequence do
      if options.tagTestSet then
        
        local sourceWord = {}
        
        for word in string.gmatch(DataAsItIs[pairIterate][i],"[^\t]+") do
          table.insert(sourceWord, word)
        end
        
        --f:write(sourceWord[1].."\t"..tagReverseVocabulary[predictedSequence[i]]:upper().."\t"..sourceWord[2].."\t"..sourceWord[3])
        f:write(sourceWord[1].."\t"..sourceWord[2].."\t"..tagReverseVocabulary[predictedSequence[i]]:upper())
        f:write("\n")
      else
        f:write(DataAsItIs[pairIterate][i]:upper().."\t")
        
        cm:add(predictedSequence[i], target[i])
        
  --Do we have word embeddings for that word or not
        if linePresent[i] == 1 then
          cmKnown:add(predictedSequence[i], target[i])
          if predictedSequence[i] == target[i] then
            presentAndCorrect = presentAndCorrect +1
          else
            isKnownWordError = true
            presentAndWrong = presentAndWrong + 1
          end
        else
          cmUnknown:add(predictedSequence[i], target[i])
          if predictedSequence[i] == target[i] then
            absentAndCorrect = absentAndCorrect +1
          else
            absentAndWrong = absentAndWrong + 1
          end
        end
        
        f:write(tagReverseVocabulary[predictedSequence[i]]:upper())
        f:write("\n")
      end
    end
    
    f:write("\n")
  end
    
  
  f:close()
  
  if not options.tagTestSet then
    print("Total Words Statistics")
    print(cm)
    cm:zero()
  end
  
  if not options.tagTestSet then
    print("Known Words Statistics")
    print(cmKnown)
    cmKnown:zero()
  end
  
  if not options.tagTestSet then
    print("Unknown Words Statistics")
    print(cmUnknown)
    cmUnknown:zero()
  end
  
  if not options.tagTestSet then
    print("Words Present and correct ".. presentAndCorrect)
    print("Words Present and wrong ".. presentAndWrong)
    print("Words Absent and correct ".. absentAndCorrect)
    print("Words Absent and wrong ".. absentAndWrong)
  end
  
end



function train(Epoch, cos, mEpoch)
  local previousCost = cos
  
  local epoch = Epoch
  
  local maxEpoch = 200
  
  if mEpoch ~= nil then
    maxEpoch = mEpoch
  end
  
--Repeat over complete train dataset as many times as maxEpoch
  while epoch < maxEpoch do
    print("==> doing epoch "..epoch.." on training data with eta :"..optimState.learningRate)
    nClock = os.clock() 
    
    memory:remember('both') 
    memory:training()
    memory:forget()
    
    memory1:remember('both') 
    memory1:training()
    memory1:forget()
    
    for eachSentence =1,#trainDataSource do
--For every sentence
      if mEpoch == nil then
        xlua.progress(eachSentence, #trainDataSource)
      end
      
      local feval = function(params_)
        local loss = 0
      
        if params_ ~= parameters then
            parameters:copy(params_)
        end
        grad_params:zero()
  
        loss = loss + performTraining(trainDataSource[eachSentence],trainDataTags[eachSentence], extractCharacterInput(trainDataCharacter[eachSentence]))
        
        grad_params:div(#trainDataSource[eachSentence])
        
-- clip gradient element-wise
        grad_params:clamp(-5, 5)
    
        return loss, grad_params
      end
      optimMethod(feval, parameters, optimState)
        
      if eachSentence%1000 == 0 then
        collectgarbage()
      end
    end
    
    local cost = 0.0
    
    print("Elapsed time: " .. os.clock()-nClock)
    nClock = os.clock() 
    
    cost = validation(tuneDataSource, tuneDataTags, tuneDataCharacter)
    
    print("Elapsed time: " .. os.clock()-nClock)
    
    if epoch == 1 then
      previousCost = cost
      
      local filename = "output/"..options.language.."/"..options.modelInfo.."/optimState_"..epoch.."_"..options.hiddenSize
      os.execute('mkdir -p ' .. sys.dirname(filename))
      torch.save(filename, optimState)
      
      filename = "output/"..options.language.."/"..options.modelInfo.."/topLayer_"..epoch.."_"..options.hiddenSize
      torch.save(filename, sequenceLabeler)
      
      filename = "output/"..options.language.."/"..options.modelInfo.."/memory_"..epoch.."_"..options.hiddenSize
      torch.save(filename, memory)
      
      filename = "output/"..options.language.."/"..options.modelInfo.."/memory1_"..epoch.."_"..options.hiddenSize
      torch.save(filename, memory1)
            
      epoch = epoch + 1
    else
      if cost >= previousCost then
       
        local previousLR = optimState.learningRate
        
        local filename = "output/"..options.language.."/"..options.modelInfo.."/optimState_"..(epoch-1).."_"..options.hiddenSize
        optimState = torch.load(filename)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/topLayer_"..(epoch-1).."_"..options.hiddenSize
        sequenceLabeler = torch.load(filename)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/memory_"..(epoch-1).."_"..options.hiddenSize
        memory = torch.load(filename)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/memory1_"..(epoch-1).."_"..options.hiddenSize
        memory1 = torch.load(filename)
        
        parameters, grad_params = nn.Container():add(sequenceLabeler):add(memory):add(memory1):getParameters()
        
        optimState.learningRate = previousLR * 0.7
      	if optimState.learningRate <= 2e-3 then       
          saveModel()
          test(testDataSource, testDataTags, testDataAsItIs, (epoch-1), testWordPresent, testDataCharacter, testcharacterAsItIs)
      		os.exit(-2)
      	end
      else
        previousCost = cost

        if epoch >= 2 then
        local filename = "output/"..options.language.."/"..options.modelInfo.."/optimState_"..(epoch-1).."_"..options.hiddenSize
        os.remove(filename)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/topLayer_"..(epoch-1).."_"..options.hiddenSize
        os.remove(filename)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/memory_"..(epoch-1).."_"..options.hiddenSize
        os.remove(filename)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/memory1_"..(epoch-1).."_"..options.hiddenSize
        os.remove(filename)
    end

        local filename = "output/"..options.language.."/"..options.modelInfo.."/optimState_"..epoch.."_"..options.hiddenSize
        torch.save(filename, optimState)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/topLayer_"..epoch.."_"..options.hiddenSize
        torch.save(filename, sequenceLabeler)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/memory_"..epoch.."_"..options.hiddenSize
        torch.save(filename, memory)
        
        filename = "output/"..options.language.."/"..options.modelInfo.."/memory1_"..epoch.."_"..options.hiddenSize
        torch.save(filename, memory1)
    
        epoch = epoch + 1
      end
    end
    
  end
  return cost
end

function readModel()
  local filename = options.unsupPath.."/embed_"..options.hiddenSize
  embed = torch.load(filename)

  for i = minCharacternGramsequence,maxCharacternGramsequence do
    filename = options.unsupPath.."/charLearnerL1_"..i.."_"..options.hiddenSize
    charLearner[i-minCharacternGramsequence +1] = torch.load(filename)
  end

  filename = options.unsupPath.."/f-lstm_"..options.hiddenSize
  lstm = torch.load(filename)

  filename = options.unsupPath.."/b-lstm_"..options.hiddenSize
  lstm1 = torch.load(filename)

  if options.useGPU then
    memory1:cuda()
    memory:cuda()
    sequenceLabeler:cuda()
    criterion:cuda()
  end

  parameters, grad_params = nn.Container():add(sequenceLabeler):add(memory):add(memory1):getParameters()
end

function saveModel()
  local filename = "output/"..options.language.."/"..options.modelInfo.."/embed_"..options.hiddenSize
  torch.save(filename,embed)

  for i = minCharacternGramsequence,maxCharacternGramsequence do
    filename = "output/"..options.language.."/"..options.modelInfo.."/charLearnerL1_"..i.."_"..options.hiddenSize
    torch.save(filename, charLearner[i-minCharacternGramsequence +1])
  end

  filename = "output/"..options.language.."/"..options.modelInfo.."/f-lstm_"..options.hiddenSize
  torch.save(filename, lstm)

  filename = "output/"..options.language.."/"..options.modelInfo.."/b-lstm_"..options.hiddenSize
  torch.save(filename, lstm1)

end


if options.predict then 
  print("On Prediction Mode")
  
  local nClock = os.clock()

  local filename = "output/"..options.language.."/"..options.modelInfo.."/optimState_"..options.preLoadModule.."_"..options.hiddenSize
  optimState = torch.load(filename)
  
  filename = "output/"..options.language.."/"..options.modelInfo.."/topLayer_"..options.preLoadModule.."_"..options.hiddenSize
  sequenceLabeler = torch.load(filename)
  
  filename = "output/"..options.language.."/"..options.modelInfo.."/memory_"..options.preLoadModule.."_"..options.hiddenSize
  memory = torch.load(filename)

  filename = "output/"..options.language.."/"..options.modelInfo.."/memory1_"..options.preLoadModule.."_"..options.hiddenSize
  memory1 = torch.load(filename)

  print("Elapsed time: " .. os.clock()-nClock)

  print("Finished Loading Model")
  
  nClock = os.clock()
  
  parameters, grad_params = nn.Container():add(sequenceLabeler):add(memory):add(memory1):getParameters()
  
  print("Elapsed time: " .. os.clock()-nClock)
  print("Finished Combining Model")
 
  saveModel()
  test(testDataSource, testDataTags, testDataAsItIs, options.preLoadModule, testWordPresent, testDataCharacter, testcharacterAsItIs)
  
else
  local epoch = 0
  local previousCost = 0.0
  
  if options.intermediate then
    
    epoch = options.preLoadModule
    local filename = "output/"..options.language.."/"..options.modelInfo.."/optimState_"..options.preLoadModule.."_"..options.hiddenSize
    optimState = torch.load(filename)
    optimState.learningRate = options.lr
    
    filename = "output/"..options.language.."/"..options.modelInfo.."/topLayer_"..options.preLoadModule.."_"..options.hiddenSize
    sequenceLabeler = torch.load(filename)
    
    filename = "output/"..options.language.."/"..options.modelInfo.."/memory_"..options.preLoadModule.."_"..options.hiddenSize
    memory = torch.load(filename)
    
    filename = "output/"..options.language.."/"..options.modelInfo.."/memory1_"..options.preLoadModule.."_"..options.hiddenSize
    memory1 = torch.load(filename)
    
    if options.useGPU then
      memory1:cuda()
      memory:cuda()
      sequenceLabeler:cuda()
      criterion:cuda()
    end
  
    parameters, grad_params = nn.Container():add(sequenceLabeler):add(memory):add(memory1):getParameters()
  
  end
  
  previousCost = validation(tuneDataSource, tuneDataTags, tuneDataCharacter)
  train( epoch+1,  previousCost )   
end
