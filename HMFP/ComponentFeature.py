import numpy as np
from pymatgen.core import Composition, Structure, Element
from matminer.featurizers.composition import ElementFraction, BandCenter


# calculate the fraction of Li of a composition
def Li_fraction(composition):
    """
    :param structure: a string of composition
    :return: one dimensional matrix of atomic number and fraction
    """
    composition = Composition(composition)
    symbol = composition.elements
    dic_metal={}
    dic_nonmetal={}
    for i in range(len(symbol)):
        if symbol[i].is_metal:
            dic_metal[str(symbol[i])]=int(symbol[i].number)
        else:
            dic_nonmetal[str(symbol[i])]=int(symbol[i].number)
    #print()
    dic_metal = sorted(dic_metal.items(), key=lambda kv:kv[1])
    dic_nonmetal = sorted(dic_nonmetal.items(), key=lambda kv:kv[1])
    dic = dic_metal+dic_nonmetal
    #print(dic[0][0])
    symbol=[]
    for i in range(len(dic)):
        a = Element(str(dic[i][0]))
        symbol.append(a)
    #print(symbol)
    EF = ElementFraction()
    site_feature = np.zeros(12)
    index = 0
    for i in range(len(symbol)):
        if index > 0:
            if not symbol[i].is_metal and index <= 6:
                index = 6 # start index of non_metal
        site_feature[index] = symbol[i].number
        site_feature[index+1] = EF.featurize(composition)[symbol[i].number-1]
        index += 2
    return site_feature


# calculate the band center by first ionization energy and electron affinity
def band_center(composition):
    composition = Composition(composition)
    BC = BandCenter()
    bandcenter = BC.featurize(composition)
    print(bandcenter)
    return bandcenter


def main():
    for i in range(1,10):
        structure = Structure.from_file('../cif/Li/'+str(i)+'.cif')
        try:
            a = band_center(structure.composition)
        except:
            print(i)


if __name__ == '__main__':
    main()
