# max_polys from: http://users.ece.cmu.edu/~koopman/lfsr/


def get_primitive_polys(size):
    outpolys = []
    f = open(f'./primitive_polys/{size}.txt', "r")
    polys = f.readlines()
    f.close()

    for poly in polys:
        hexpoly = int(poly.rstrip(), 16)
        binpoly = str(bin(hexpoly))[2:]
        binpoly = [x for x in binpoly]
        binpoly.reverse()
        outpoly = []
        for i in range(0, len(binpoly)):
            if binpoly[i] == "1":
                outpoly.append(i+1)
        outpoly.reverse()
        outpolys.append(outpoly)

    return outpolys
