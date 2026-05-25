#include "bezierCurve.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Dense>

static double squaredDistance(const orderedEdgePoint &a, const orderedEdgePoint &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

static double minDistanceToBezier(const std::vector<orderedEdgePoint> &controlPoints,
                                  const orderedEdgePoint &point)
{
    if (controlPoints.empty())
        return 0.0;

    const int samples = 32;
    double bestT = 0.0;
    double bestD2 = squaredDistance(evaluateBezier(controlPoints, 0.0), point);

    for (int i = 1; i <= samples; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(samples);
        double d2 = squaredDistance(evaluateBezier(controlPoints, t), point);
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestT = t;
        }
    }

    const double step = 1.0 / static_cast<double>(samples);
    double a = std::max(0.0, bestT - step);
    double b = std::min(1.0, bestT + step);

    // Local refinement with golden-section search over the best sample interval.
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a);
    double d = a + gr * (b - a);
    double fc = squaredDistance(evaluateBezier(controlPoints, c), point);
    double fd = squaredDistance(evaluateBezier(controlPoints, d), point);

    for (int iter = 0; iter < 16; ++iter)
    {
        if (fc < fd)
        {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = squaredDistance(evaluateBezier(controlPoints, c), point);
        }
        else
        {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = squaredDistance(evaluateBezier(controlPoints, d), point);
        }
    }

    const double minD2 = std::min(fc, fd);
    return std::sqrt(minD2);
}

std::vector<double> computeBezierResiduals(const std::vector<orderedEdgePoint> &controlPoints, const Edge &edge)
{
    const size_t n = edge.mvPoints.size();
    std::vector<double> residuals;
    residuals.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
        const orderedEdgePoint &edgePoint = edge.mvPoints[i];
        residuals.push_back(minDistanceToBezier(controlPoints, edgePoint));
    }

    return residuals;
}

orderedEdgePoint evaluateBezier(const std::vector<orderedEdgePoint> &controlPoints, double t)
{
    int order = controlPoints.size() - 1;
    orderedEdgePoint point(0.0, 0.0);
    for (int i = 0; i <= order; ++i)
    {
        // double binomialCoeff = 1.0;
        // for (int j = 0; j < i; ++j)
        //     binomialCoeff *= (order - j) / static_cast<double>(j + 1);

        // double weight = binomialCoeff * std::pow(1 - t, order - i) * std::pow(t, i);
        double weight = bernsteinBasis(order, i, t);
        point.x += weight * controlPoints[i].x;
        point.y += weight * controlPoints[i].y;
    }
    return point;
}

double normalCdf(double x)
{
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double normalInv(double p)
{
    // Acklam's inverse normal approximation.
    const double a1 = -3.969683028665376e+01;
    const double a2 = 2.209460984245205e+02;
    const double a3 = -2.759285104469687e+02;
    const double a4 = 1.383577518672690e+02;
    const double a5 = -3.066479806614716e+01;
    const double a6 = 2.506628277459239e+00;

    const double b1 = -5.447609879822406e+01;
    const double b2 = 1.615858368580409e+02;
    const double b3 = -1.556989798598866e+02;
    const double b4 = 6.680131188771972e+01;
    const double b5 = -1.328068155288572e+01;

    const double c1 = -7.784894002430293e-03;
    const double c2 = -3.223964580411365e-01;
    const double c3 = -2.400758277161838e+00;
    const double c4 = -2.549732539343734e+00;
    const double c5 = 4.374664141464968e+00;
    const double c6 = 2.938163982698783e+00;

    const double d1 = 7.784695709041462e-03;
    const double d2 = 3.224671290700398e-01;
    const double d3 = 2.445134137142996e+00;
    const double d4 = 3.754408661907416e+00;

    if (p <= 0.0)
        return -DBL_MAX;
    if (p >= 1.0)
        return DBL_MAX;

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;
    double q = 0.0;
    double r = 0.0;

    if (p < plow)
    {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p > phigh)
    {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }

    q = p - 0.5;
    r = q * q;
    return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double shapiroWilkPValue(const std::vector<double> &samples)
{
    const size_t n = samples.size();
    if (n < 3)
        return 0.0;

    std::vector<double> x = samples;
    std::sort(x.begin(), x.end());

    double mean = 0.0;
    for (double v : x)
        mean += v;
    mean /= static_cast<double>(n);

    double ssd = 0.0;
    for (double v : x)
    {
        double d = v - mean;
        ssd += d * d;
    }
    if (ssd <= std::numeric_limits<double>::epsilon())
        return 0.0;

    const size_t n2 = n / 2;
    std::vector<double> m(n2);
    for (size_t i = 0; i < n2; ++i)
    {
        double p = (static_cast<double>(i) + 1.0 - 0.375) / (static_cast<double>(n) + 0.25);
        m[i] = normalInv(p);
    }

    double m2 = 0.0;
    for (double v : m)
        m2 += v * v;
    const double scale = 1.0 / std::sqrt(m2);
    for (double &v : m)
        v *= scale;

    double num = 0.0;
    for (size_t i = 0; i < n2; ++i)
        num += m[i] * (x[n - 1 - i] - x[i]);

    double w = (num * num) / ssd;
    if (w >= 1.0)
        w = 1.0 - 1e-16;
    if (w <= 0.0)
        return 0.0;

    const double y = std::log(1.0 - w);
    double mu = 0.0;
    double sigma = 1.0;

    if (n <= 11)
    {
        const double n1 = static_cast<double>(n);
        const double n2v = n1 * n1;
        const double n3v = n2v * n1;
        mu = -0.0006714 * n3v + 0.025054 * n2v - 0.39978 * n1 + 0.5440;
        sigma = std::exp(-0.0020322 * n3v + 0.062767 * n2v - 0.77857 * n1 + 1.3822);
    }
    else
    {
        const double ln = std::log(static_cast<double>(n));
        const double ln2 = ln * ln;
        const double ln3 = ln2 * ln;
        mu = -1.5861 - 0.31082 * ln - 0.083751 * ln2 + 0.0038915 * ln3;
        sigma = std::exp(-0.4803 - 0.082676 * ln + 0.0030302 * ln2);
    }

    const double z = (y - mu) / sigma;
    const double p = 1.0 - normalCdf(z);
    return std::clamp(p, 0.0, 1.0);
}

std::vector<std::vector<orderedEdgePoint>> fitBezierAdaptive(const std::vector<orderedEdgePoint> &points, double rho_p)
{
    if (points.size() < 10)
        return std::vector<std::vector<orderedEdgePoint>>();

    Edge localEdge;
    localEdge.mvPoints = points;

    std::vector<double> residuals;
    std::vector<orderedEdgePoint> lastControlPoints;

    for (int order = 1; order <= 3; ++order)
    {
        lastControlPoints = fitBezierWithEndPoints(localEdge, order);
        if (lastControlPoints.empty())
            return std::vector<std::vector<orderedEdgePoint>>();

        residuals = computeBezierResiduals(lastControlPoints, localEdge);
        double maxResidual = *std::max_element(residuals.begin(), residuals.end());

        if (maxResidual < rho_p) {
            // printf("Order %d: Max Residual = %.4f\n", order, maxResidual);
            return {lastControlPoints};
        }

        // if (maxResidual < rho_p && shapiroWilkPValue(residuals) > 0.05) {
        //     // printf("Order %d: Max Residual = %.4f\n", order, maxResidual);
        //     return {lastControlPoints};
        // }
    }

    if (residuals.empty())
        return std::vector<std::vector<orderedEdgePoint>>();

    auto maxIt = std::max_element(residuals.begin(), residuals.end());
    size_t splitIndex = static_cast<size_t>(std::distance(residuals.begin(), maxIt));

    if (splitIndex == 0 || splitIndex >= points.size() - 1)
        return {lastControlPoints};

    std::vector<orderedEdgePoint> points1(points.begin(), points.begin() + splitIndex + 1);
    std::vector<orderedEdgePoint> points2(points.begin() + splitIndex, points.end());

    std::vector<std::vector<orderedEdgePoint>> curves1 = fitBezierAdaptive(points1, rho_p);
    std::vector<std::vector<orderedEdgePoint>> curves2 = fitBezierAdaptive(points2, rho_p);

    curves1.insert(curves1.end(), curves2.begin(), curves2.end());
    return curves1;
}

std::vector<orderedEdgePoint> fitBezierWithEndPoints(const Edge &edge, int order)
{
    int n = edge.mvPoints.size();
    if (n < 2 || order < 1)
        return std::vector<orderedEdgePoint>();
    
    orderedEdgePoint startPoint = edge.mvPoints[0];
    orderedEdgePoint endPoint = edge.mvPoints[n - 1];

    if (order == 1)
        return {startPoint, endPoint};

    int unknowns = order - 1;
    Eigen::MatrixXd A(n, unknowns);
    Eigen::VectorXd b_x(n);
    Eigen::VectorXd b_y(n);

    for (int i = 0; i < n; ++i)
    {
        double t = static_cast<double>(i) / (n - 1);

        double B_0 = bernsteinBasis(order, 0, t);
        double B_end = bernsteinBasis(order, order, t);

        double known_x = B_0 * startPoint.x + B_end * endPoint.x;
        double known_y = B_0 * startPoint.y + B_end * endPoint.y;

        for (int k = 1; k <= unknowns; ++k)
        {
            A(i, k - 1) = bernsteinBasis(order, k, t);
        }

        const orderedEdgePoint &q_i = edge.mvPoints[i];
        b_x(i) = q_i.x - known_x;
        b_y(i) = q_i.y - known_y;
    }

    Eigen::VectorXd middle_x = A.colPivHouseholderQr().solve(b_x);
    Eigen::VectorXd middle_y = A.colPivHouseholderQr().solve(b_y);

    std::vector<orderedEdgePoint> controlPoints;
    controlPoints.reserve(order + 1);
    controlPoints.push_back(startPoint);

    for (int k = 0; k < unknowns; ++k)
    {
        controlPoints.emplace_back(middle_x(k), middle_y(k));
    }

    controlPoints.push_back(endPoint);
    return controlPoints;
}

double bernsteinBasis(int n, int i, double t)
{
    double binomialCoeff = 1.0;
    for (int j = 0; j < i; ++j)
        binomialCoeff *= (n - j) / static_cast<double>(j + 1);

    return binomialCoeff * std::pow(1 - t, n - i) * std::pow(t, i);
}
