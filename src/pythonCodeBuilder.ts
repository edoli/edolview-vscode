/* Original python code
import socket
import json
from struct import pack
import importlib.util
import numpy as np
import zlib

class EdolView:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send_image(self, name:str, image, float_to_half, extra={}):
        # convert torch to numpy array
        if type(image) != np.ndarray:
            torch_spec = importlib.util.find_spec('torch')

            if torch_spec is not None:
                import torch

                if type(image) == torch.Tensor:
                    if hasattr(image, 'detach'):
                        image = image.detach()

                    if hasattr(image, 'cpu'):
                        image = image.cpu()
                        
                    if hasattr(image, 'numpy'):
                        image = image.numpy()
                        
        if type(image) != np.ndarray:
            raise Exception('image should be np.ndarray')
        
        initial_shape = image.shape
        dtype = image.dtype
        
        # Try to convert (B, C, H, W) or (C, H, W) to (H, W, C)
        while len(image.shape) > 3:
            image = image[0, ...]

        if len(image.shape) == 2:
            image = image[None, ...]

        if image.shape[-1] > 4:
            image = image.transpose(1, 2, 0)        
            
        if image.shape[-1] > 4:
            raise Exception('image dimension not valid shape: ' + str(initial_shape))

        # Convert to PNG if cv2 is installed and image is integer. Otherwise use zlib compress
        cv2_spec = importlib.util.find_spec('cv2')
            
        if np.issubdtype(dtype, np.integer) and cv2_spec is not None:
            import cv2

            retval, buf = cv2.imencode('.png', image[:, :, ::-1])
            buf_bytes = buf.tobytes()

            extra['compression'] = 'png'
        else:
            if (image.dtype == np.float32 or image.dtype == np.float64) and float_to_half:
                image = image.astype(np.float16)

            if not image.data.c_contiguous:
                image = image.copy()

            buf_bytes = zlib.compress(image.data)

            extra['compression'] = 'zlib'

        extra['nbytes'] = image.nbytes
        extra['shape'] = image.shape
        extra['dtype'] = image.dtype.name

        extra_str = json.dumps(extra)

        # Encode string and send to socket
        name_bytes = name.encode('utf-8')
        extra_bytes = extra_str.encode('utf-8')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))

            s.send(pack('!i', len(name_bytes)))
            s.send(pack('!i', len(extra_bytes)))
            s.send(pack('!i', len(buf_bytes)))
            s.send(name_bytes)
            s.send(extra_bytes)
            s.sendall(buf_bytes)
            s.close()
*/

const pythonCode = `
M=Exception
I=hasattr
H=type
E=len
import socket as F,json
from struct import pack as G
import importlib.util,numpy as C,zlib
class EdolView:
	def __init__(A,host,port):A.host=host;A.port=port
	def send_image(N,name,image,float_to_half,extra={}):
		R='utf-8';Q='compression';L='!i';K=None;D=extra;A=image
		if H(A)!=C.ndarray:
			S=importlib.util.find_spec('torch')
			if S is not K:
				import torch
				if H(A)==torch.Tensor:
					if I(A,'detach'):A=A.detach()
					if I(A,'cpu'):A=A.cpu()
					if I(A,'numpy'):A=A.numpy()
		if H(A)!=C.ndarray:raise M('image should be np.ndarray')
		T=A.shape;U=A.dtype
		while E(A.shape)>3:A=A[0,...]
		if E(A.shape)==2:A=A[K,...]
		if A.shape[-1]>4:A=A.transpose(1,2,0)
		if A.shape[-1]>4:raise M('image dimension not valid shape: '+str(T))
		V=importlib.util.find_spec('cv2')
		if C.issubdtype(U,C.integer)and V is not K:import cv2;Y,W=cv2.imencode('.png',A[:,:,::-1]);J=W.tobytes();D[Q]='png'
		else:
			if(A.dtype==C.float32 or A.dtype==C.float64)and float_to_half:A=A.astype(C.float16)
			if not A.data.c_contiguous:A=A.copy()
			J=zlib.compress(A.data);D[Q]='zlib'
		D['nbytes']=A.nbytes;D['shape']=A.shape;D['dtype']=A.dtype.name;X=json.dumps(D);O=name.encode(R);P=X.encode(R)
		with F.socket(F.AF_INET,F.SOCK_STREAM)as B:B.connect((N.host,N.port));B.send(G(L,E(O)));B.send(G(L,E(P)));B.send(G(L,E(J)));B.send(O);B.send(P);B.sendall(J);B.close()
`;

const pythonCodeBuilder = (evaluateName: string, host: string, port: number, floatToHalf: boolean) => {
    const evaluateNameEscape = evaluateName.replaceAll('\'', '\\\'').replaceAll('\"', '\\\"');
    const floatToHalfStr = floatToHalf ? "True" : "False";

    return `
${pythonCode}

EdolView(host='${host}', port=${port}).send_image('${evaluateNameEscape}', ${evaluateName}, ${floatToHalfStr})
    `;
};

export { pythonCodeBuilder };