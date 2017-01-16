local data = {}

function data.getData(trainSize, seqLength, n, nt, lt)
   -- Generate random sequence of n different possible values
   local s = torch.Tensor(trainSize):random(n)

   -- Labels for training
   local y = torch.ones(trainSize)

   for i = seqLength, trainSize do
      if          s[i-5] == 1
              and s[i-4] == 7
              and s[i-3] == 3
              and s[i-2] == 5
              and s[i-1] == 5
              and s[i] == lt then
         y[i] = 2
      end
   end

   -- Mapping of input from R trainSize to trainSize x 2
   -- a => 1 => <1, 0>
   -- b => 2 => <0, 1>
   -- (one hot encoding)

   local x = torch.zeros(trainSize, nt) -- leave space for target alphabet size
   for i = 1, trainSize do
      for j = 1, n do
         if s[i] == j then
            x[i][j] = 1
         end
      end
   end
   return x, y
end

return data