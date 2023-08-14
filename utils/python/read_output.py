import h5py
import numpy as np

def collectResults(format, nSample):
    filenames = [format % k for k in range(nSample)]
    
    results = np.array((nSample, 5), dtype=float)
    attrs = ["rom_assemble", "rom_solve", "fom_assemble", "fom_solve", "rel_error"]
    
    for k, filename in enumerate(filenames):
        with h5py.File(filename, 'r') as f:
            for a, attr in enumerate(attrs):
                results[k, a]   = f.attrs[attr]
    
    results = np.sort(results)

    for a, attr in enumerate(attrs):
        print("================================")
        print(attr)
        print("median: %.5E" % results[int(nSample * 0.5), a])
        print("minimum: %.5E" % results[0, a])
        print("maxinum: %.5E" % results[-1, a])
        print("-1std: %.5E" % results[int(nSample * (0.5 - 0.34)), a])
        print("+1std: %.5E" % results[int(nSample * (0.5 + 0.34)), a])
        print("================================")