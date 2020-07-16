""" 
Lattice sequence generator from Magic Point Shop (https://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/)

Adapted from https://bitbucket.org/dnuyens/qmc-generators/src/master/python/latticeseq_b2.py

Reference:
    
    [1] F.Y. Kuo & D. Nuyens.
    Application of quasi-Monte Carlo methods to elliptic PDEs with random diffusion coefficients 
    - a survey of analysis and implementation, Foundations of Computational Mathematics, 
    16(6):1631-1696, 2016.
    springer link: https://link.springer.com/article/10.1007/s10208-016-9329-5
    arxiv link: https://arxiv.org/abs/1606.06613
    
    [2] D. Nuyens, `The Magic Point Shop of QMC point generators and generating
    vectors.` MATLAB and Python software, 2018. Available from
    https://people.cs.kuleuven.be/~dirk.nuyens/

"""

from numpy import arange, array, outer, log2, vstack, double, floor, ceil, zeros

"""
generating vector from
    Constructing embedded lattice rules for multivariate integration
    R Cools, FY Kuo, D Nuyens -  SIAM J. Sci. Comput., 28(6), 2162-2188.
maximum number of points was set to 2**20, maximum number of dimensions is 250
constructed for unanchored Sobolev space with order dependent weights of order 2,
meaning that all 2-dimensional projections are taken into account explicitly
(in this case all choices of weights are equivalent and this is thus a generic order 2 rule)
"""
exod2_base2_m20_CKN_z = array([
    1, 433461, 315689, 441789, 501101, 146355, 88411, 215837, 273599, 151719, 258185, 357967, 96407, 
    203741, 211709, 135719, 100779, 85729, 14597, 94813, 422013, 484367, 355029, 123065, 467905, 41129, 
    298607, 375981, 256421, 279695, 164795, 256413, 267543, 505211, 225547, 50293, 97031, 86633, 203383, 
    427981, 221421, 465833, 329843, 212325, 467017, 214065, 98063, 128867, 63891, 426443, 244641, 56441, 
    357107, 199459, 169327, 407687, 154961, 64579, 436713, 322855, 435589, 220821, 72219, 344125, 315189, 
    105979, 421183, 212659, 26699, 491987, 310515, 344337, 443019, 174213, 244609, 5979, 85677, 148663, 
    514069, 172383, 238589, 458305, 460201, 487365, 454835, 452035, 55005, 517221, 85841, 434641, 387469, 
    24883, 154373, 145103, 416491, 252109, 509385, 296473, 248789, 297219, 119711, 252395, 188293, 23943, 
    264817, 242005, 26689, 51931, 490263, 155451, 365301, 445277, 311581, 306887, 331445, 208941, 385313, 
    307593, 359113, 67919, 351803, 335955, 326111, 57853, 52153, 84863, 158013, 272483, 419143, 252581, 
    372097, 177007, 145815, 350453, 412791, 435559, 387627, 35887, 48461, 389563, 68569, 118715, 250699, 
    183713, 29615, 168429, 292527, 86465, 450915, 239063, 23051, 347131, 138885, 243505, 201835, 269831, 
    265457, 496089, 273459, 276803, 225507, 148131, 87909, 115693, 45749, 3233, 194661, 329135, 90215, 
    104003, 27611, 437589, 422687, 19029, 284433, 348413, 289359, 418785, 293911, 358343, 85919, 501439, 
    462941, 301185, 292875, 242667, 408165, 137921, 329199, 308125, 48743, 122291, 362643, 90781, 448407, 
    25389, 78793, 362423, 239423, 280833, 55483, 43757, 138415, 395119, 175965, 253391, 462987, 50655, 67155, 
    142149, 314277, 452523, 364029, 323001, 105873, 231785, 329547, 517581, 64375, 180745, 30693, 321739, 259327, 
    523313, 123863, 446629, 112611, 134019, 442879, 516621, 469677, 271077, 83859, 195209, 385581, 3287, 261841, 
    16525, 243831, 505215, 37669, 275001, 118849, 475943, 56509, 239489, 35893, 31015, 458209, 292255, 94197, 279055, 
    7573, 233705, 339587, 396313, 310037, 371939, 494279, 261481, 2875, 51129, 204067, 40633, 459101, 226639, 89795, 
    464665, 439937, 388665, 277539, 370801, 438367, 73733, 166153, 200849, 250477, 148655, 445817, 375723, 373433, 
    154819, 367247, 462549, 382217, 269073, 15985, 206263, 507895, 335263, 251183, 236851, 285491, 371291, 20143, 
    471543, 334263, 397501, 52335, 122837, 160981, 332741, 341961, 320455, 144133, 410489, 440261, 274789, 83793, 
    353867, 310001, 161271, 28267, 400007, 469779, 351385, 158419, 301117, 234521, 260047, 312511, 213851, 332001, 
    3699, 518163, 119209, 329387, 149889, 485193, 505407, 326067, 149541, 102343, 441707, 499551, 501199, 77817, 
    355999, 128165, 396261, 247463, 9733, 481107, 411379, 479917, 84085, 380091, 489765, 504237, 47847, 496129, 
    343905, 496621, 498123, 270835, 459931, 314289, 89077, 505051, 11647, 26765, 349111, 357217, 493937, 179089, 
    300189, 143621, 205639, 244475, 303281, 180189, 70443, 301471, 17853, 17121, 243179, 377849, 209079, 167565, 
    357373, 309503, 367039, 136041, 247861, 226573, 63631, 344345, 256401, 138305, 271675, 354845, 420971, 442981, 
    225321, 342755, 427957, 493767, 488177, 141063, 224621, 9439, 217623, 242451, 508557, 379609, 202291, 266555, 
    452509, 379789, 89867, 519873, 163115, 237191, 235291, 149683, 187821, 508801, 425951, 239141, 284505, 498919, 
    493857, 97373, 92147, 492967, 302591, 225277, 16947, 275043, 322807, 377713, 408445, 187103, 185133, 505963, 
    386109, 96301, 470963, 407939, 6601, 409277, 5031, 128747, 393271, 415197, 114049, 223999, 99373, 482183, 
    504981, 295837, 34235, 40765, 408397, 216741, 422925, 496079, 300813, 277283, 312489, 368009, 161369, 362997, 
    6663, 509953, 387903, 97597, 238917, 378851, 190545, 430029, 204931, 466553, 293441, 327939, 183495, 463331, 
    422655, 428099, 20715, 477503, 465937, 270399, 139589, 129581, 215571, 299645, 125221, 23345, 229345, 138059, 
    521769, 14731, 318159, 190173, 361381, 485577, 512807, 268009, 185937, 210939, 86965, 113005, 296923, 85753, 
    381527, 196325, 274565, 182689, 200951, 117371, 489747, 19521, 426587, 168393, 486039, 220941, 392473, 344051, 
    412275, 501127, 434941, 85569, 406757, 371643, 470783, 466117, 170707, 473019, 494155, 411809, 13371, 202745, 
    23597, 25621, 64351, 508445, 204947, 38279, 264269, 230499, 405605, 68513, 414481, 301849, 6815, 406425, 62881, 
    174349, 505503, 329037, 104357, 113815, 137669, 181689, 493057, 296191, 135279, 236891, 82135, 371269, 483993, 
    394407, 372929, 139823, 114515, 416815, 260309, 489593, 156763, 21523, 189285, 308129, 155369, 213557, 298023, 
    391439, 379245, 409109, 229765, 28521, 464087, 470911, 435965, 201451, 64371, 370499, 276377, 331635, 196813, 
    379415, 229547, 430067, 137053, 312839, 390385, 77155, 163911, 514381, 487453],
    dtype=double)
exod2_len = len(exod2_base2_m20_CKN_z)

def mps_lattice_gen(n_min, n_max, d):
    """
    Generate d dimensionsal lattice samples from n_min to n_max
    
    Args:
        d (int): dimension of the problem, 1<=d<=100.
        n_min (int): minimum index. Must be 0 or n_max/2
        n_max (int): maximum index (not inclusive)
    """
    if n_min==n_max:
        return array([],dtype=double)
    if d > exod2_len:
        raise Exception('MPS Lattice has max dimensions %d'%exod2_len)
    if n_max > 2**20:
        raise Exception('MPS Lattice has maximum points 2^20')
    z = exod2_base2_m20_CKN_z[:d]
    m_low = floor(log2(n_min))+1 if n_min > 0 else 0
    m_high = ceil(log2(n_max))
    gen_block = lambda n: (outer(arange(1, n+1, 2), z) % n) / float(n)
    x_lat_full = vstack([gen_block(2**m) for m in range(int(m_low),int(m_high)+1)])
    cut1 = int(floor(n_min-2**(m_low-1))) if n_min>0 else 0
    cut2 = int(cut1+n_max-n_min)
    x_lat = x_lat_full[cut1:cut2,:]
    return x_lat
