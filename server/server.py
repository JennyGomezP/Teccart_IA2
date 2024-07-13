from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from distances import retrieve_similar_image
import numpy as np

loaded_signatures_glcm = np.load('./server/glcm.npy')
loaded_signatures_bitdesc = np.load('./server/bitdesc.npy')

#creacion de FastAPI App
app = FastAPI()

class Feature(BaseModel):
    features: list
    descriptor: str
    distances: str
    num_result: int
    
#deficion de middleware
@app.post('/similarity')
async def similarImage(feat_list: Feature):
    print(f'Image features: {feat_list.features}')
    print(f'Descriptor: {feat_list.descriptor}')
    print(f'Distance: {feat_list.distances}')
    print(f'Numero Resultat: {feat_list.num_result}')
    try:
        if (feat_list.descriptor == 'glcm'):
            results = retrieve_similar_image(loaded_signatures_glcm, feat_list.features, feat_list.distances, feat_list.num_result)
        else:
            results = retrieve_similar_image(loaded_signatures_bitdesc, feat_list.features, feat_list.distances, feat_list.num_result)
        response = {'similar_image': results}
        response = JSONResponse(content=response)
        return response
    except Exception as e:
        print(f'Error:', (e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8881)