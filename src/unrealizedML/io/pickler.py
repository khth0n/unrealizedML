import io
import torch
import pickle

def pickle_tensor(tensor: torch.Tensor, filename: str) -> None:

    buffer = io.BytesIO()
    #torch.save(tensor, buffer)
    
    with open(filename, 'wb') as file:
        
        torch.save(tensor, file)
        #pickle.dump(buffer.getvalue(), file)
    
def unpickle_tensor(filename: str, dtype: torch.dtype) -> torch.Tensor:
    
    with open(filename, 'rb') as file:
        
        return torch.frombuffer(
            pickle.load(file),
            dtype=dtype
        )
    
torch.manual_seed(0)
test = torch.rand((1, 2, 2, 2), dtype=torch.float32)

print(test)

torch.save(test, 'test_tensor.pkl')

unpickled = torch.load('test_tensor.pkl')
print(unpickled)