BIGWIG_BACKEND='pyBigWig'

def set_bigwig_backend(backend):
    global BIGWIG_BACKEND
    
    assert backend in ['pyBigWig', 'memmap'], "Backend options are pyBigWig or memmap!"

    BIGWIG_BACKEND = backend
