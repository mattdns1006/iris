models = {}

function models.model1()
	local model = nn.Sequential()
	model:add(nn.Linear(4,10))
	model:add(nn.Tanh())
	model:add(nn.Linear(10,3))
	model:add(nn.SoftMax())
	return model
end

return models
	


