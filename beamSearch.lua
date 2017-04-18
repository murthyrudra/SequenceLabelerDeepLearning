require 'nn'
require 'torch'
local class = require 'class'

  local Hypothesis = class("Hypothesis")
  
  function Hypothesis:__init()
    self.targetOutput = {}
    self.probabilityScore = 0.0
    self.currentCost = 0.0
    self.output = {}
    self.predictions = {}
  end
  
  function Hypothesis:setPredictions(predict)
    for i = 1,#predict do
      table.insert(self.predictions, predict[i]:clone())
    end
  end
  
  function Hypothesis:createPredictions(predict)
    table.insert(self.predictions, predict:clone())
  end
  
  function Hypothesis:getPredictions()
    return self.predictions
  end
  
  function Hypothesis:createTargetWord(targetWord)
    table.insert(self.targetOutput, targetWord)
  end
  
  function Hypothesis:copyTargetWord(targetWord)
    for i = 1,#targetWord do
      table.insert(self.targetOutput, targetWord[i])
    end
  end
  
  function Hypothesis:setTargetWord(targetWord)
    self.output[1] = targetWord
  end
  
  function Hypothesis:getTargetIndex()
    return self.output[1]
  end
  
  function Hypothesis:getTarget()
    return self.targetOutput
  end
  
  function Hypothesis:insertProbability(probability)
    self.probabilityScore = probability
  end
  
  function Hypothesis:getProbability()
    return self.probabilityScore
  end
  
  function Hypothesis:insertCost(cost, previousStateCost)
    self.currentCost = previousStateCost + cost
  end
  
  function Hypothesis:getCost()
    return self.currentCost 
  end
  
function doBeamSearch(options, memoryActivations, numberOfTags,  useGPU, inputSize, sequenceLabeler)
  -- Initially the predicted target word is nil
  local init_predictions = torch.zeros(numberOfTags)
  
  --get logSoftMax Predictions
  local prePredictions
  
  if options.useGPU then
    local tempInput
    
    tempInput = torch.cat(memoryActivations[1]:clone():float(), init_predictions:float()):cuda()
    prePredictions = sequenceLabeler:forward(tempInput)
  else
    local tempInput
    
    tempInput = torch.cat(memoryActivations[1]:clone():float(), init_predictions:float())
    prePredictions = sequenceLabeler:forward(tempInput)
  end
  
  -- Use every output as first prediction to build the next set of predictions
  local setOfHypothesis = {}
  for i=1,numberOfTags do
    -- Create a initial hypothesis for every target word
    local tempHypothesis = Hypothesis.new()
    -- target word is identified by it's index
    tempHypothesis:createTargetWord(i)
    tempHypothesis:setTargetWord(i)
    tempHypothesis:insertProbability(prePredictions[i])
    tempHypothesis:insertCost(prePredictions[i],0.0)
--    tempHypothesis:createPredictions(prePredictions)
    
    -- insert the new hypothesis into the set of base hypothesis
    table.insert(setOfHypothesis,tempHypothesis)
  end
  
  table.sort(setOfHypothesis, function(hypo1, hypo2) return hypo1:getCost() > hypo2:getCost() end)
  

  local newHypothesis = {}
  local currentTableLength = 0
   
  for iterate = 2,inputSize do
  -- no hypothesis generated initially
    newHypothesis = {}
    
    -- for every hypothesis in the base set of hypothesis
    for i =1, #setOfHypothesis do
      local hypothesisPairs = setOfHypothesis[i]
      
      local getPrevIndex = hypothesisPairs:getTargetIndex()
      local previousOutput = torch.zeros(numberOfTags)
      previousOutput[getPrevIndex] = 1.0
      
      local inputToLayer = torch.cat(memoryActivations[iterate]:clone():float(), previousOutput:float()):clone()
      
      local prePredictions_new
      if options.useGPU then
        prePredictions_new = sequenceLabeler:forward(inputToLayer:cuda())
      else
        prePredictions_new = sequenceLabeler:forward(inputToLayer)
      end
      
      -- For every target word generate a new hypothesis
      for j=1,numberOfTags do
        -- add the list of already generated words          
        local tempHypothesis = Hypothesis.new()
        local t = hypothesisPairs:getTarget()
        for ii = 1,#t do
          tempHypothesis:createTargetWord(t[ii])
        end
        
        tempHypothesis:createTargetWord(j)
        tempHypothesis:setTargetWord(j)
        
        tempHypothesis:insertProbability(prePredictions_new[j])
        tempHypothesis:insertCost(prePredictions_new[j], hypothesisPairs:getCost())
--        tempHypothesis:setPredictions(hypothesisPairs:getPredictions())
--        tempHypothesis:createPredictions(prePredictions_new)
        
        table.insert(newHypothesis,tempHypothesis)
      end
    end
    
    -- Need to prune out hypothesis based on beams size
    -- sort the hypothesis in descending order of the log-probability values
    table.sort(newHypothesis, function(hypo1, hypo2) return hypo1:getCost() > hypo2:getCost() end)
    
    local currentTableLength = 0
    setOfHypothesis = {}
    
    while currentTableLength < numberOfTags and currentTableLength < #newHypothesis do
      local newHypothesisTemp = Hypothesis.new()
      newHypothesisTemp:copyTargetWord(newHypothesis[currentTableLength +1]:getTarget())
      newHypothesisTemp:setTargetWord(newHypothesis[currentTableLength +1]:getTargetIndex())
      newHypothesisTemp:insertProbability(newHypothesis[currentTableLength +1]:getProbability())
      newHypothesisTemp:insertCost(newHypothesis[currentTableLength +1]:getCost(),0.0)
      
--      newHypothesisTemp:setPredictions(newHypothesis[currentTableLength +1]:getPredictions())
      
      table.insert(setOfHypothesis, newHypothesisTemp)
      currentTableLength = currentTableLength + 1  
    end
    
    newHypothesis = {}
  end
  
  local predictedSequence = {}
  predictedSequence = setOfHypothesis[1]:getTarget()
  
  return predictedSequence, setOfHypothesis[1]:getCost()
  
end

function doBeamSearchNew(options, memoryActivations, numberOfTags,  useGPU, inputSize, sequenceLabeler, embeddingsDirect)
  -- Initially the predicted target word is nil
  local init_predictions = torch.zeros(numberOfTags)
  
  --get logSoftMax Predictions
  local prePredictions
  
  if options.useGPU then
    local tempInput
    
    tempInput = torch.cat(memoryActivations[1]:clone():float(), init_predictions:float()):cuda()
    local tempOut = torch.cat(tempInput:float():clone(), embeddingsDirect[1]:float()):clone()
    prePredictions = sequenceLabeler:forward(tempOut:cuda())
  else
    local tempInput
    
    tempInput = torch.cat(memoryActivations[1]:clone():float(), init_predictions:float())
    local tempOut = torch.cat(tempInput:float():clone(), embeddingsDirect[1]:float()):clone()
    prePredictions = sequenceLabeler:forward(tempOut)
  end
  
  -- Use every output as first prediction to build the next set of predictions
  local setOfHypothesis = {}
  for i=1,numberOfTags do
    -- Create a initial hypothesis for every target word
    local tempHypothesis = Hypothesis.new()
    -- target word is identified by it's index
    tempHypothesis:createTargetWord(i)
    tempHypothesis:setTargetWord(i)
    tempHypothesis:insertProbability(prePredictions[i])
    tempHypothesis:insertCost(prePredictions[i],0.0)
--    tempHypothesis:createPredictions(prePredictions)
    
    -- insert the new hypothesis into the set of base hypothesis
    table.insert(setOfHypothesis,tempHypothesis)
  end
  
  table.sort(setOfHypothesis, function(hypo1, hypo2) return hypo1:getCost() > hypo2:getCost() end)
  

  local newHypothesis = {}
  local currentTableLength = 0
   
  for iterate = 2,inputSize do
  -- no hypothesis generated initially
    newHypothesis = {}
    
    -- for every hypothesis in the base set of hypothesis
    for i =1, #setOfHypothesis do
      local hypothesisPairs = setOfHypothesis[i]
      
      local getPrevIndex = hypothesisPairs:getTargetIndex()
      local previousOutput = torch.zeros(numberOfTags)
      previousOutput[getPrevIndex] = 1.0
      
      local inputToLayer = torch.cat(memoryActivations[iterate]:clone():float(), previousOutput:float()):clone()
      
      local tempOut = torch.cat(inputToLayer:clone(), embeddingsDirect[iterate]:float()):clone()
      
    
      local prePredictions_new
      if options.useGPU then
        prePredictions_new = sequenceLabeler:forward(tempOut:cuda())
      else
        prePredictions_new = sequenceLabeler:forward(tempOut)
      end
      
      -- For every target word generate a new hypothesis
      for j=1,numberOfTags do
        -- add the list of already generated words          
        local tempHypothesis = Hypothesis.new()
        local t = hypothesisPairs:getTarget()
        for ii = 1,#t do
          tempHypothesis:createTargetWord(t[ii])
        end
        
        tempHypothesis:createTargetWord(j)
        tempHypothesis:setTargetWord(j)
        
        tempHypothesis:insertProbability(prePredictions_new[j])
        tempHypothesis:insertCost(prePredictions_new[j], hypothesisPairs:getCost())
--        tempHypothesis:setPredictions(hypothesisPairs:getPredictions())
--        tempHypothesis:createPredictions(prePredictions_new)
        
        table.insert(newHypothesis,tempHypothesis)
      end
    end
    
    -- Need to prune out hypothesis based on beams size
    -- sort the hypothesis in descending order of the log-probability values
    table.sort(newHypothesis, function(hypo1, hypo2) return hypo1:getCost() > hypo2:getCost() end)
    
    local currentTableLength = 0
    setOfHypothesis = {}
    
    while currentTableLength < numberOfTags and currentTableLength < #newHypothesis do
      local newHypothesisTemp = Hypothesis.new()
      newHypothesisTemp:copyTargetWord(newHypothesis[currentTableLength +1]:getTarget())
      newHypothesisTemp:setTargetWord(newHypothesis[currentTableLength +1]:getTargetIndex())
      newHypothesisTemp:insertProbability(newHypothesis[currentTableLength +1]:getProbability())
      newHypothesisTemp:insertCost(newHypothesis[currentTableLength +1]:getCost(),0.0)
      
--      newHypothesisTemp:setPredictions(newHypothesis[currentTableLength +1]:getPredictions())
      
      table.insert(setOfHypothesis, newHypothesisTemp)
      currentTableLength = currentTableLength + 1  
    end
    
    newHypothesis = {}
  end
  
  local predictedSequence = {}
  predictedSequence = setOfHypothesis[1]:getTarget()
  
  return predictedSequence, setOfHypothesis[1]:getCost()
  
end
