models = {}

function models.model1()
	local model = nn.Sequential()
	model:add(nn.Linear(4,10))
	model:add(nn.Tanh())
	model:add(nn.Linear(10,3))
	model:add(nn.SoftMax())
	return model
end

function models.model2()
	local model = nn.Sequential()
	model:add(nn.Dropout(params.dropout))
	c = nn.ConcatTable()
	c:add(nn.Linear(4,10))
	c:add(nn.Linear(4,5))
	model:add(c)

	p1 = nn.ParallelTable()
	p1:add(nn.Tanh())
	p1:add(nn.Tanh())
	model:add(p1)

	p2 = nn.ParallelTable()
	p2:add(nn.Linear(10,10))
	p2:add(nn.Linear(5,5))
	model:add(p2)
	
	model:add(nn.JoinTable(2))
	model:add(nn.Tanh())
	model:add(nn.Linear(15,3))
	model:add(nn.Tanh())
	model:add(nn.Linear(3,3))
	model:add(nn.SoftMax())
	return model

end

return models
	


