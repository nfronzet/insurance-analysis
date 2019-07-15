import numpy

def cos_get_weights(cos, credentials, filename):
    
    print('Attempting to load previously saved weights...')
    try:
        saved_weights = cos.get_object(Bucket=credentials['BUCKET'], Key=filename)['Body'].read().decode('utf-8')
        
        w_list = []

        lines = saved_weights.split('\n')
        lines.pop()
        
        for line in lines:
            tensor = []
            vals = iter(line.split())
            dims = []
            dim = int(next(vals))
            val = next(vals)
            while dim > 0:
                dims.append(int(val))
                val = next(vals)
                dim = dim - 1
            n = numpy.prod(dims)
            while n > 0:
                try:
                    tensor.append(float(val))
                    val = next(vals)
                    n = n - 1
                except:
                    break #¯\_(ツ)_/¯
            tensor = numpy.reshape(tensor,dims)
            w_list.append(tensor)
            
        print('Weights file found on Cloud Object Storage')
        return w_list
    except:
        raise FileNotFoundError()
        
def cos_save_weights(model, cos, credentials, filename):
    new_weights = model.get_weights()
    w_buf = ''
    print('Saving model weights to Cloud Storage...')
    for tensor in new_weights:
        dims = tensor.shape
        w_buf += str(len(dims))
        w_buf += ' '
        for dim in dims:
            w_buf += str(dim)
            w_buf += ' '
        length = numpy.prod(dims)
        t_arr = numpy.ndarray.reshape(tensor,length)
        for n in t_arr:
            w_buf += str(n)
            w_buf += ' '
        w_buf += ' \n'
    
    try:
        cos.put_object(Body=w_buf, Bucket=credentials['BUCKET'], Key=filename)
        print('Model weights saved successfully')
    except:
        raise ConnectionError()
