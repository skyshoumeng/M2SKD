import numpy as np

def ground_h(ground, inst):
    '''
    ground NX3
    inst MX3
    
    reutrn 对齐后的inst表示(z坐标已经经过对齐), 如果无法对齐，则返回None
    '''
    
    hmean, hstd = np.mean(ground, axis=2), np.std(ground, axis=2)
    if hstd > 0.3:
        return None
    
    sort_z = np.sort(inst[:,2])
    shape = sort_z.shape[0]
    
    assert np.round(shape/4) - np.round(shape/20) > 0
    
    align_height = np.mean(sort_z[np.round(shape/20):np.round(shape/4)]) - hmean + 0.05
    inst[:,2] -= align_height
    
    return inst
    
    
    
    