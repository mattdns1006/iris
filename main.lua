require "torch"
require "nn"
require "nngraph"
require "cunn"
require "cutorch"
require "gnuplot"
require "optim"
dofile("oneHotEncode.lua")
shuffle = require "shuffle"
models = require "models"

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training iris')
cmd:text()
cmd:text('Options')
cmd:option('-lr',0.01,'Learning rate')
cmd:option('-momentum',0.9,'Momentum')
cmd:option('-batchSize',1,'batchSize')
cmd:option('-cuda',1,'CUDA')
cmd:text()

params = cmd:parse(arg)
params.rundir = cmd:string('experiment', params, {dir=true})

model = models.model1()
--criterion = nn.CrossEntropyCriterion()
criterion = nn.MSECriterion()
batchSize = params.batchSize 

optimMethod = optim.sgd
optimState = {
	learningRate = params.lr,
	weightDecay = 0,
	momentum = params.momentum,
	learningRateDecay = 0
}
--print("==> Optimizer parameters ", optimState)

-- Data
local irisLoader = require "irisLoader.lua"
local data = irisLoader.load_data()
X,y,names = data.inputs, data.targets, data.targets_by_name
y = oneHotInts(y,3)
nObs = #names

-- Cuda 
if params.cuda == 1 then 
	model = model:cuda()
	criterion = criterion:cuda()
	X = X:cuda()
	y = y:cuda()
end

-- Split into train test
split = 110 
trainX ,testX  = X[{{1,split}}], X[{{split+1,X:size()[1]}}]
trainy ,testy  = y[{{1,split}}], y[{{split+1,y:size()[1]}}]

-- Reshuffle fn 
function reshuffle()
	local randomPerm = torch.randperm(nTrain):long()
	trainX = trainX:index(1,randomPerm)
	trainy = trainy:index(1,randomPerm)
end

-- Training
parameters, gradParameters = model:getParameters()
function feval(x)
		if x ~= parameters then parameters:copy(x) end
		
		gradParameters:zero()
		outputs = model:forward(inputs)
		loss = criterion:forward(outputs, targets)
		dLoss_dOutput = criterion:backward(outputs,targets)

		model:backward(inputs,dLoss_dOutput)
		
		return loss, gradParameters
end

function trainEpoch()
	local epochLoss = {}
	nTrain = trainX:size()[1]
	reshuffle()
	local from = 1
	for i = 1,  nTrain, batchSize do 
		local from = i  
		local to = math.min(from + batchSize - 1,nTrain)
		inputs = trainX[{{from,to}}]
		targets = trainy[{{from,to}}]
		_, batchLoss = optim.sgd(feval,parameters,optimState)
		epochLoss[#epochLoss + 1] = batchLoss
	end
	local epochLossT = torch.Tensor{epochLoss}
	return epochLossT:mean()
end

function testEpoch()
	local epochLoss = {}
	local nTest = testX:size()[1]
	local from = 1
	for i = 1,  nTest, batchSize do 
		local from = i  
		local to = math.min(from + batchSize - 1,nTest)
		local inputs = testX[{{from,to}}]
		local targets = testy[{{from,to}}]
		local predictions = model:forward(inputs)
		local batchLoss = criterion:forward(predictions,targets)
		epochLoss[#epochLoss + 1] = batchLoss
	end
	local epochLossT = torch.Tensor{epochLoss}
	return epochLossT:mean()
end

function train()
	epoch = 0
	trainEpochLosses = {}
	testEpochLosses = {}
	while true do 
		print("==> Epoch number " .. epoch)
		trainTime = torch.Timer()
		trainEpochLosses[#trainEpochLosses + 1] = trainEpoch()
		print("==> Time taken for epoch ".. trainTime:time().real .. " secs.")
		epoch = epoch + 1 
		testEpochLosses[#testEpochLosses + 1] = testEpoch()

		local function printLosses()
			print("==> Train, test epoch losses ")
			print(trainEpochLosses[#trainEpochLosses])
			print(testEpochLosses[#testEpochLosses])
		end
		printLosses()

		x = torch.range(1,#trainEpochLosses)
		trainEpochLossesT = torch.Tensor{trainEpochLosses}:view(-1)
		testEpochLossesT = torch.Tensor{testEpochLosses}:view(-1)
		if epoch % 100 == 0 then
			gnuplot.plot({'train',x,trainEpochLossesT},{'test',x,testEpochLossesT})
		end
	end
end
--train()




