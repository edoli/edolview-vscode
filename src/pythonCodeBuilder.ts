/* Original python code
import socket
import json
from struct import pack
import importlib.util
import numpy as np
import zlib

def area_interpolate(im, scale):
    new_h = im.shape[0] // scale
    new_w = im.shape[1] // scale

    clip_h = new_h * scale
    clip_w = new_w * scale

    buf = np.zeros((new_h, new_w, im.shape[2]), dtype=np.float32)

    for i in range(scale):
        for j in range(scale):
            buf += im[i:clip_h:scale, j:clip_w:scale]

    buf = (buf / (scale * scale)).astype(im.dtype)

    return buf

class EdolView:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send_image(self, name:str, image, float_to_half, downscale_factor=1, extra={}):
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

        if downscale_factor != 1:
            image = area_interpolate(image, downscale_factor)

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
K=Exception
J=range
I=hasattr
H=type
E=len
import socket as F,json
from struct import pack as G
import importlib.util,numpy as B,zlib
def P(im,scale):
	C=im;A=scale;E=C.shape[0]//A;F=C.shape[1]//A;G=E*A;H=F*A;D=B.zeros((E,F,C.shape[2]),dtype=B.float32)
	for I in J(A):
		for K in J(A):D+=C[I:G:A,K:H:A]
	D=(D/(A*A)).astype(C.dtype);return D
class EdolView:
	def __init__(A,host,port):A.host=host;A.port=port
	def send_image(L,name:str,image,float_to_half,downscale_factor=1,extra={}):
		M=downscale_factor;D=extra;A=image
		if H(A)!=B.ndarray:
			Q=importlib.util.find_spec('torch')
			if Q is not None:
				import torch
				if H(A)==torch.Tensor:
					if I(A,'detach'):A=A.detach()
					if I(A,'cpu'):A=A.cpu()
					if I(A,'numpy'):A=A.numpy()
		if H(A)!=B.ndarray:raise K('image should be np.ndarray')
		R=A.shape;S=A.dtype
		while E(A.shape)>3:A=A[0,...]
		if E(A.shape)==2:A=A[None,...]
		if A.shape[-1]>4:A=A.transpose(1,2,0)
		if A.shape[-1]>4:raise K('image dimension not valid shape: '+str(R))
		if M!=1:A=P(A,M)
		T=importlib.util.find_spec('cv2')
		if B.issubdtype(S,B.integer)and T is not None:import cv2;W,U=cv2.imencode('.png',A[:,:,::-1]);J=U.tobytes();D['compression']='png'
		else:
			if(A.dtype==B.float32 or A.dtype==B.float64)and float_to_half:A=A.astype(B.float16)
			if not A.data.c_contiguous:A=A.copy()
			J=zlib.compress(A.data);D['compression']='zlib'
		D['nbytes']=A.nbytes;D['shape']=A.shape;D['dtype']=A.dtype.name;V=json.dumps(D);N=name.encode('utf-8');O=V.encode('utf-8')
		with F.socket(F.AF_INET,F.SOCK_STREAM)as C:C.connect((L.host,L.port));C.send(G('!i',E(N)));C.send(G('!i',E(O)));C.send(G('!i',E(J)));C.send(N);C.send(O);C.sendall(J);C.close()
`;

const pythonCodeBuilder = (evaluateName: string, host: string, port: number, floatToHalf: boolean, downscale: number) => {
    const evaluateNameEscape = evaluateName.replaceAll('\'', '\\\'').replaceAll('\"', '\\\"');
    const floatToHalfStr = floatToHalf ? "True" : "False";
    const downscaleStr = downscale.toString();

    return `
${pythonCode}

EdolView(host='${host}', port=${port}).send_image('${evaluateNameEscape}', ${evaluateName}, ${floatToHalfStr}, ${downscaleStr})
    `;
};

export { pythonCodeBuilder };