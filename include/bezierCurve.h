#ifndef BEZIERCURVE_H
#define BEZIERCURVE_H

#include <vector>

#include "edge.h"

// Your bezier curve class definition here
std::vector<double> computeBezierResiduals(const std::vector<orderedEdgePoint>& controlPoints, const Edge& edge);
orderedEdgePoint evaluateBezier(const std::vector<orderedEdgePoint>& controlPoints, double t);
std::vector<std::vector<orderedEdgePoint>> fitBezierAdaptive(const std::vector<orderedEdgePoint> &points, double rho_p = 10);
std::vector<orderedEdgePoint> fitBezierWithEndPoints(const Edge& edge, int order);
double bernsteinBasis(int n, int i, double t);

#endif // BEZIERCURVE_H