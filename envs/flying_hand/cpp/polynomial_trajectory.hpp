/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/
#pragma once

#include "root_finder.hpp"

#include <Eigen/Eigen>

#include <cfloat>
#include <cmath>
#include <iostream>
#include <vector>

namespace flygripper_common {
template <int Dim, int Order> class Piece {
public:
  using CoefficientMat = Eigen::Matrix<double, Dim, Order + 1>;
  using VelCoefficientMat = Eigen::Matrix<double, Dim, Order>;
  using AccCoefficientMat = Eigen::Matrix<double, Dim, Order - 1>;
  using VectorMat = Eigen::Matrix<double, Dim, 1>;

private:
  double duration;
  CoefficientMat coeffMat;

public:
  Piece() = default;

  Piece(double dur, const CoefficientMat &cMat)
      : duration(dur), coeffMat(cMat) {}

  inline int getDim() const { return Dim; }

  inline int getDegree() const { return Order; }

  inline double getDuration() const { return duration; }

  inline const CoefficientMat &getCoeffMat() const { return coeffMat; }

  inline VectorMat getPos(const double &t) const {
    VectorMat pos = VectorMat::Zero();
    double tn = 1.0;
    for (int i = Order; i >= 0; i--) {
      pos += tn * coeffMat.col(i);
      tn *= t;
    }
    return pos;
  }

  inline VectorMat getVel(const double &t) const {
    VectorMat vel = VectorMat::Zero();
    double tn = 1.0;
    int n = 1;
    for (int i = Order - 1; i >= 0; i--) {
      vel += n * tn * coeffMat.col(i);
      tn *= t;
      n++;
    }
    return vel;
  }

  inline VectorMat getAcc(const double &t) const {
    VectorMat acc = VectorMat::Zero();
    double tn = 1.0;
    int m = 1;
    int n = 2;
    for (int i = Order - 2; i >= 0; i--) {
      acc += m * n * tn * coeffMat.col(i);
      tn *= t;
      m++;
      n++;
    }
    return acc;
  }

  inline VectorMat getJer(const double &t) const {
    VectorMat jer = VectorMat::Zero();
    double tn = 1.0;
    int l = 1;
    int m = 2;
    int n = 3;
    for (int i = Order - 3; i >= 0; i--) {
      jer += l * m * n * tn * coeffMat.col(i);
      tn *= t;
      l++;
      m++;
      n++;
    }
    return jer;
  }

  inline VectorMat getSnap(const double &t) const {
    VectorMat snap = VectorMat::Zero();
    snap.setZero();
    double tn = 1.0;
    int l = 1;
    int m = 2;
    int n = 3;
    int o = 4;
    for (int i = Order - 4; i >= 0; --i) {
      snap += o * l * m * n * tn * coeffMat.col(i);
      tn *= t;
      ++l;
      ++m;
      ++n;
      ++o;
    }
    return snap;
  }

  inline CoefficientMat normalizePosCoeffMat() const {
    CoefficientMat nPosCoeffsMat;
    double t = 1.0;
    for (int i = Order; i >= 0; i--) {
      nPosCoeffsMat.col(i) = coeffMat.col(i) * t;
      t *= duration;
    }
    return nPosCoeffsMat;
  }

  inline VelCoefficientMat normalizeVelCoeffMat() const {
    VelCoefficientMat nVelCoeffMat;
    int n = 1;
    double t = duration;
    for (int i = Order - 1; i >= 0; i--) {
      nVelCoeffMat.col(i) = n * coeffMat.col(i) * t;
      t *= duration;
      n++;
    }
    return nVelCoeffMat;
  }

  inline AccCoefficientMat normalizeAccCoeffMat() const {
    AccCoefficientMat nAccCoeffMat;
    int n = 2;
    int m = 1;
    double t = duration * duration;
    for (int i = Order - 2; i >= 0; i--) {
      nAccCoeffMat.col(i) = n * m * coeffMat.col(i) * t;
      n++;
      m++;
      t *= duration;
    }
    return nAccCoeffMat;
  }

  inline double getMaxVelRate() const {
    VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
    Eigen::VectorXd coeff(2 * nVelCoeffMat.cols() - 1);
    for (int i = 0; i < Dim; i++) {
      coeff += RootFinder::polySqr(nVelCoeffMat.row(i));
    }
    int N = coeff.size();
    int n = N - 1;
    for (int i = 0; i < N; i++) {
      coeff(i) *= n;
      n--;
    }
    if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON) {
      return getVel(0.0).norm();
    } else {
      double l = -0.0625;
      double r = 1.0625;
      while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON) {
        l = 0.5 * l;
      }
      while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON) {
        r = 0.5 * (r + 1.0);
      }
      std::set<double> candidates = RootFinder::solvePolynomial(
          coeff.head(N - 1), l, r, FLT_EPSILON / duration);
      candidates.insert(0.0);
      candidates.insert(1.0);
      double maxVelRateSqr = -INFINITY;
      double tempNormSqr;
      for (std::set<double>::const_iterator it = candidates.begin();
           it != candidates.end(); it++) {
        if (0.0 <= *it && 1.0 >= *it) {
          tempNormSqr = getVel((*it) * duration).squaredNorm();
          maxVelRateSqr =
              maxVelRateSqr < tempNormSqr ? tempNormSqr : maxVelRateSqr;
        }
      }
      return sqrt(maxVelRateSqr);
    }
  }

  inline double getMaxAccRate() const {
    AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
    Eigen::VectorXd coeff(2 * nAccCoeffMat.cols() - 1);
    for (int i = 0; i < Dim; i++) {
      coeff += RootFinder::polySqr(nAccCoeffMat.row(i));
    }
    int N = coeff.size();
    int n = N - 1;
    for (int i = 0; i < N; i++) {
      coeff(i) *= n;
      n--;
    }
    if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON) {
      return getAcc(0.0).norm();
    } else {
      double l = -0.0625;
      double r = 1.0625;
      while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON) {
        l = 0.5 * l;
      }
      while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON) {
        r = 0.5 * (r + 1.0);
      }
      std::set<double> candidates = RootFinder::solvePolynomial(
          coeff.head(N - 1), l, r, FLT_EPSILON / duration);
      candidates.insert(0.0);
      candidates.insert(1.0);
      double maxAccRateSqr = -INFINITY;
      double tempNormSqr;
      for (std::set<double>::const_iterator it = candidates.begin();
           it != candidates.end(); it++) {
        if (0.0 <= *it && 1.0 >= *it) {
          tempNormSqr = getAcc((*it) * duration).squaredNorm();
          maxAccRateSqr =
              maxAccRateSqr < tempNormSqr ? tempNormSqr : maxAccRateSqr;
        }
      }
      return sqrt(maxAccRateSqr);
    }
  }

  inline bool checkMaxVelRate(const double &maxVelRate) const {
    double sqrMaxVelRate = maxVelRate * maxVelRate;
    if (getVel(0.0).squaredNorm() >= sqrMaxVelRate ||
        getVel(duration).squaredNorm() >= sqrMaxVelRate) {
      return false;
    } else {
      VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
      Eigen::VectorXd coeff(2 * nVelCoeffMat.cols() - 1);
      for (int i = 0; i < Dim; i++) {
        coeff += RootFinder::polySqr(nVelCoeffMat.row(i));
      }
      double t2 = duration * duration;
      coeff.tail<1>()(0) -= sqrMaxVelRate * t2;
      return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;
    }
  }

  inline bool checkMaxAccRate(const double &maxAccRate) const {
    double sqrMaxAccRate = maxAccRate * maxAccRate;
    if (getAcc(0.0).squaredNorm() >= sqrMaxAccRate ||
        getAcc(duration).squaredNorm() >= sqrMaxAccRate) {
      return false;
    } else {
      AccCoefficientMat nAccCoeffMat = normalizeAccCoeffMat();
      Eigen::VectorXd coeff(2 * nAccCoeffMat.cols() - 1);
      for (int i = 0; i < Dim; i++) {
        coeff += RootFinder::polySqr(nAccCoeffMat.row(i));
      }
      double t2 = duration * duration;
      double t4 = t2 * t2;
      coeff.tail<1>()(0) -= sqrMaxAccRate * t4;
      return RootFinder::countRoots(coeff, 0.0, 1.0) == 0;
    }
  }
};

template <int Dim, int Order> class PolynomialTrajectory {
public:
  using PieceType = Piece<Dim, Order>;
  using Pieces = std::vector<PieceType>;
  using CoefficientMats = std::vector<typename PieceType::CoefficientMat>;
  using VectorMats =
      Eigen::Matrix<double, Dim, Eigen::Dynamic>; // dynamic is Piece Num

private:
  Pieces pieces;

public:
  PolynomialTrajectory() = default;

  PolynomialTrajectory(const std::vector<double> &durs,
                       const CoefficientMats &cMats) {
    int N = std::min(durs.size(), cMats.size());
    pieces.reserve(N);
    for (int i = 0; i < N; i++) {
      pieces.emplace_back(durs[i], cMats[i]);
    }
  }

  inline int getPieceNum() const { return pieces.size(); }

  inline Eigen::VectorXd getDurations() const {
    int N = getPieceNum();
    Eigen::VectorXd durations(N);
    for (int i = 0; i < N; i++) {
      durations(i) = pieces[i].getDuration();
    }
    return durations;
  }

  inline double getTotalDuration() const {
    int N = getPieceNum();
    double totalDuration = 0.0;
    for (int i = 0; i < N; i++) {
      totalDuration += pieces[i].getDuration();
    }
    return totalDuration;
  }

  inline VectorMats getPositions() const {
    int N = getPieceNum();
    VectorMats positions(Dim, N + 1);
    for (int i = 0; i < N; i++) {
      positions.col(i) = pieces[i].getCoeffMat().col(Order);
    }
    positions.col(N) = pieces[N - 1].getPos(pieces[N - 1].getDuration());
    return positions;
  }

  inline const PieceType &operator[](int i) const { return pieces[i]; }

  inline PieceType &operator[](int i) { return pieces[i]; }

  inline void clear(void) {
    pieces.clear();
    return;
  }

  inline typename Pieces::const_iterator begin() const {
    return pieces.begin();
  }

  inline typename Pieces::const_iterator end() const { return pieces.end(); }

  inline typename Pieces::iterator begin() { return pieces.begin(); }

  inline typename Pieces::iterator end() { return pieces.end(); }

  inline void reserve(const int &n) {
    pieces.reserve(n);
    return;
  }

  inline void emplace_back(const PieceType &piece) {
    pieces.emplace_back(piece);
    return;
  }

  inline void emplace_back(const double &dur,
                           const typename PieceType::CoefficientMat &cMat) {
    pieces.emplace_back(dur, cMat);
    return;
  }

  inline void append(const PolynomialTrajectory<Dim, Order> &traj) {
    pieces.insert(pieces.end(), traj.begin(), traj.end());
    return;
  }

  inline int locatePieceIdx(double &t) const {
    int N = getPieceNum();
    int idx;
    double dur;
    for (idx = 0; idx < N && t > (dur = pieces[idx].getDuration()); idx++) {
      t -= dur;
    }
    if (idx == N) {
      idx--;
      t += pieces[idx].getDuration();
    }
    return idx;
  }

  inline Eigen::MatrixXd getPVAJS(double t) const {
    int pieceIdx = locatePieceIdx(t);
    Eigen::MatrixXd pvajs(Dim, 5);
    auto &piece = pieces[pieceIdx];
    pvajs.col(0) = piece.getPos(t);
    pvajs.col(1) = piece.getVel(t);
    pvajs.col(2) = piece.getAcc(t);
    pvajs.col(3) = piece.getJer(t);
    pvajs.col(4) = piece.getSnap(t);
    return pvajs;
  }

  inline typename PieceType::VectorMat getPos(double t) const {
    int pieceIdx = locatePieceIdx(t);
    return pieces[pieceIdx].getPos(t);
  }

  inline typename PieceType::VectorMat getVel(double t) const {
    int pieceIdx = locatePieceIdx(t);
    return pieces[pieceIdx].getVel(t);
  }

  inline typename PieceType::VectorMat getAcc(double t) const {
    int pieceIdx = locatePieceIdx(t);
    return pieces[pieceIdx].getAcc(t);
  }

  inline typename PieceType::VectorMat getJer(double t) const {
    int pieceIdx = locatePieceIdx(t);
    return pieces[pieceIdx].getJer(t);
  }

  inline typename PieceType::VectorMat getSnap(double t) const {
    int pieceIdx = locatePieceIdx(t);
    return pieces[pieceIdx].getSnap(t);
  }

  inline typename PieceType::VectorMat getJuncPos(int juncIdx) const {
    if (juncIdx != getPieceNum()) {
      return pieces[juncIdx].getCoeffMat().col(Order);
    } else {
      return pieces[juncIdx - 1].getPos(pieces[juncIdx - 1].getDuration());
    }
  }

  inline typename PieceType::VectorMat getJuncVel(int juncIdx) const {
    if (juncIdx != getPieceNum()) {
      return pieces[juncIdx].getCoeffMat().col(Order - 1);
    } else {
      return pieces[juncIdx - 1].getVel(pieces[juncIdx - 1].getDuration());
    }
  }

  inline typename PieceType::VectorMat getJuncAcc(int juncIdx) const {
    if (juncIdx != getPieceNum()) {
      return pieces[juncIdx].getCoeffMat().col(Order - 2) * 2.0;
    } else {
      return pieces[juncIdx - 1].getAcc(pieces[juncIdx - 1].getDuration());
    }
  }

  inline double getMaxVelRate() const {
    int N = getPieceNum();
    double maxVelRate = -INFINITY;
    double tempNorm;
    for (int i = 0; i < N; i++) {
      tempNorm = pieces[i].getMaxVelRate();
      maxVelRate = maxVelRate < tempNorm ? tempNorm : maxVelRate;
    }
    return maxVelRate;
  }

  inline double getMaxAccRate() const {
    int N = getPieceNum();
    double maxAccRate = -INFINITY;
    double tempNorm;
    for (int i = 0; i < N; i++) {
      tempNorm = pieces[i].getMaxAccRate();
      maxAccRate = maxAccRate < tempNorm ? tempNorm : maxAccRate;
    }
    return maxAccRate;
  }

  inline bool checkMaxVelRate(const double &maxVelRate) const {
    int N = getPieceNum();
    bool feasible = true;
    for (int i = 0; i < N && feasible; i++) {
      feasible = feasible && pieces[i].checkMaxVelRate(maxVelRate);
    }
    return feasible;
  }

  inline bool checkMaxAccRate(const double &maxAccRate) const {
    int N = getPieceNum();
    bool feasible = true;
    for (int i = 0; i < N && feasible; i++) {
      feasible = feasible && pieces[i].checkMaxAccRate(maxAccRate);
    }
    return feasible;
  }
};

} // namespace flygripper_common
