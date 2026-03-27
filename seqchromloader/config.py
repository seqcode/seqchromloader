BIGWIG_BACKEND='pyBigWig'

def set_bigwig_backend(backend):
    global BIGWIG_BACKEND
    
    assert backend in ['pyBigWig', 'pybigtools'], "Backend options are pyBigWig or pybigtools!"

    BIGWIG_BACKEND = backend
