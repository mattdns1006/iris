-- Function to one hot encode
function oneHotInts(ints,width)
	local height = ints:size()[1]
	local zeros = torch.zeros(height,width)
	local indicies = ints:view(-1,1):long()
	local oneHot = zeros:scatter(2,indicies,1)
	return oneHot
end
