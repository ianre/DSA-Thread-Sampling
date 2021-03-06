from shapely.geometry import Polygon
polygon = Polygon([(0, 0), (1, 1), (1, 0)])
area = polygon.area
print(area)
p = [
        190.88, 276.31, 189.85, 276.35, 188.56, 275.64, 187.08, 274.13, 184.64,
        270.25, 178.75, 255.25, 176.66, 245.38, 175.01, 240.09, 173.14, 239.95,
        168.98, 242.25, 166.97, 242.96, 165.39, 242.96, 163.95, 242.82, 162.52,
        241.67, 161.23, 239.38, 163.24, 233.06, 166.39, 230.04, 169.27, 228.18,
        175.73, 228.18, 176.87, 229.47, 181.9, 250.72, 186.49, 262.92, 191.23,
        272.25, 191.52, 274.84, 191.52, 275.78
      ]
listOf2ples = []
for i in range(0,len(p)-1):
    listOf2ples.append(   (p[i],p[i+1]) )
    i=i+1

newPoly = Polygon(listOf2ples)
area = newPoly.area
print(area)


print("Distance", newPoly.distance(polygon))


G_Points = [(0,0),(50,50),(100,0),(50,100)]
N_point = [(50,25),(0,24),(51,24)]
G_Poly = Polygon(G_Points)
N_Poly = Polygon(N_point)
print("Area G",G_Poly.area)
print("Area N",N_Poly.area)
print("Dist", G_Poly.distance(N_Poly))
print("Intersect?", G_Poly.intersects(N_Poly))