/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef MINCO_HPP
#define MINCO_HPP

#include <Eigen/Eigen>

#include <cmath>
#include <vector>

// Polynomial Trajectory for minco
#include "polynomial_trajectory.hpp"


namespace minco
{
// Trajectory here is defined as a polynomial trajectory class with 3 dimensions (X, Y, Z)
template <int Dim, int Order>
using Trajectory = flygripper_common::PolynomialTrajectory<Dim, Order>;

// The banded system class is used for solving
// banded linear system Ax=b efficiently.
// A is an N*N band matrix with lower band width lowerBw
// and upper band width upperBw.
// Banded LU factorization has O(N) time complexity.
class BandedSystem
{
 public:
  // The size of A, as well as the lower/upper
  // banded width p/q are needed
  inline void create(const int &n, const int &p, const int &q)
  {
    // In case of re-creating before destroying
    destroy();
    N = n;
    lowerBw = p;
    upperBw = q;
    int actualSize = N * (lowerBw + upperBw + 1);
    ptrData = new double[actualSize];
    std::fill_n(ptrData, actualSize, 0.0);
    return;
  }

  inline void destroy()
  {
    if (ptrData != nullptr) {
      delete[] ptrData;
      ptrData = nullptr;
    }
    return;
  }

 private:
  int N;
  int lowerBw;
  int upperBw;
  // Compulsory nullptr initialization here
  double *ptrData = nullptr;

 public:
  // Reset the matrix to zero
  inline void reset(void)
  {
    std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
    return;
  }

  // The band matrix is stored as suggested in "Matrix Computation"
  inline const double &operator()(const int &i, const int &j) const
  {
    return ptrData[(i - j + upperBw) * N + j];
  }

  inline double &operator()(const int &i, const int &j)
  {
    return ptrData[(i - j + upperBw) * N + j];
  }

  // This function conducts banded LU factorization in place
  // Note that NO PIVOT is applied on the matrix "A" for efficiency!!!
  inline void factorizeLU()
  {
    int iM, jM;
    double cVl;
    for (int k = 0; k <= N - 2; k++) {
      iM = std::min(k + lowerBw, N - 1);
      cVl = operator()(k, k);
      for (int i = k + 1; i <= iM; i++) {
        if (operator()(i, k) != 0.0) {
          operator()(i, k) /= cVl;
        }
      }
      jM = std::min(k + upperBw, N - 1);
      for (int j = k + 1; j <= jM; j++) {
        cVl = operator()(k, j);
        if (cVl != 0.0) {
          for (int i = k + 1; i <= iM; i++) {
            if (operator()(i, k) != 0.0) {
              operator()(i, j) -= operator()(i, k) * cVl;
            }
          }
        }
      }
    }
    return;
  }

  // This function solves Ax=b, then stores x in b
  // The input b is required to be N*m, i.e.,
  // m vectors to be solved.
  template <typename EIGENMAT>
  inline void solve(EIGENMAT &b) const
  {
    int iM;
    for (int j = 0; j <= N - 1; j++) {
      iM = std::min(j + lowerBw, N - 1);
      for (int i = j + 1; i <= iM; i++) {
        if (operator()(i, j) != 0.0) {
          b.row(i) -= operator()(i, j) * b.row(j);
        }
      }
    }
    for (int j = N - 1; j >= 0; j--) {
      b.row(j) /= operator()(j, j);
      iM = std::max(0, j - upperBw);
      for (int i = iM; i <= j - 1; i++) {
        if (operator()(i, j) != 0.0) {
          b.row(i) -= operator()(i, j) * b.row(j);
        }
      }
    }
    return;
  }

  // This function solves ATx=b, then stores x in b
  // The input b is required to be N*m, i.e.,
  // m vectors to be solved.
  template <typename EIGENMAT>
  inline void solveAdj(EIGENMAT &b) const
  {
    int iM;
    for (int j = 0; j <= N - 1; j++) {
      b.row(j) /= operator()(j, j);
      iM = std::min(j + upperBw, N - 1);
      for (int i = j + 1; i <= iM; i++) {
        if (operator()(j, i) != 0.0) {
          b.row(i) -= operator()(j, i) * b.row(j);
        }
      }
    }
    for (int j = N - 1; j >= 0; j--) {
      iM = std::max(0, j - lowerBw);
      for (int i = iM; i <= j - 1; i++) {
        if (operator()(j, i) != 0.0) {
          b.row(i) -= operator()(j, i) * b.row(j);
        }
      }
    }
    return;
  }
};

template <int Dim, int S, bool UNIFORM_T>  // (1)
class MINCO;

// MINCO for s=2 and non-uniform time
template <int Dim>
class MINCO<Dim, 2, false>
{
 public:
  MINCO() = default;
  ~MINCO() { A.destroy(); }

  int N;
  Eigen::Matrix<double, Dim, 2> headPV;
  Eigen::Matrix<double, Dim, 2> tailPV;
  BandedSystem A;
  Eigen::MatrixXd b;
  Eigen::VectorXd T1;
  Eigen::VectorXd T2;
  Eigen::VectorXd T3;

  // for outside use
  Eigen::MatrixXd gdP;
  Eigen::MatrixXd gdC;
  Eigen::VectorXd gdT;

  inline void reset(const Eigen::Matrix<double, Dim, 2> &headState,
                    const Eigen::Matrix<double, Dim, 2> &tailState,
                    const int &pieceNum)
  {
    N = pieceNum;
    headPV = headState;
    tailPV = tailState;
    A.create(4 * N, 4, 4);
    b.resize(4 * N, Dim);
    T1.resize(N);
    T2.resize(N);
    T3.resize(N);
    gdC.resize(4 * N, Dim);
    gdP.resize(Dim, N - 1);
    gdT.resize(N);
    return;
  }

  inline void generate(const Eigen::Matrix<double, Dim, -1> &inPs,
                       const Eigen::VectorXd &ts)
  {
    T1 = ts;
    T2 = T1.cwiseProduct(T1);
    T3 = T2.cwiseProduct(T1);

    A.reset();
    b.setZero();

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    b.row(0) = headPV.col(0).transpose();
    b.row(1) = headPV.col(1).transpose();

    for (int i = 0; i < N - 1; i++) {
      A(4 * i + 2, 4 * i + 2) = 2.0;
      A(4 * i + 2, 4 * i + 3) = 6.0 * T1(i);
      A(4 * i + 2, 4 * i + 6) = -2.0;
      A(4 * i + 3, 4 * i) = 1.0;
      A(4 * i + 3, 4 * i + 1) = T1(i);
      A(4 * i + 3, 4 * i + 2) = T2(i);
      A(4 * i + 3, 4 * i + 3) = T3(i);
      A(4 * i + 4, 4 * i) = 1.0;
      A(4 * i + 4, 4 * i + 1) = T1(i);
      A(4 * i + 4, 4 * i + 2) = T2(i);
      A(4 * i + 4, 4 * i + 3) = T3(i);
      A(4 * i + 4, 4 * i + 4) = -1.0;
      A(4 * i + 5, 4 * i + 1) = 1.0;
      A(4 * i + 5, 4 * i + 2) = 2.0 * T1(i);
      A(4 * i + 5, 4 * i + 3) = 3.0 * T2(i);
      A(4 * i + 5, 4 * i + 5) = -1.0;

      b.row(4 * i + 3) = inPs.col(i).transpose();
    }

    A(4 * N - 2, 4 * N - 4) = 1.0;
    A(4 * N - 2, 4 * N - 3) = T1(N - 1);
    A(4 * N - 2, 4 * N - 2) = T2(N - 1);
    A(4 * N - 2, 4 * N - 1) = T3(N - 1);
    A(4 * N - 1, 4 * N - 3) = 1.0;
    A(4 * N - 1, 4 * N - 2) = 2 * T1(N - 1);
    A(4 * N - 1, 4 * N - 1) = 3 * T2(N - 1);

    b.row(4 * N - 2) = tailPV.col(0).transpose();
    b.row(4 * N - 1) = tailPV.col(1).transpose();

    A.factorizeLU();
    A.solve(b);

    return;
  }

  inline void getTrajectory(Trajectory<Dim, 3> &traj) const
  {
    traj.clear();
    traj.reserve(N);
    for (int i = 0; i < N; i++) {
      traj.emplace_back(
          T1(i), b.block<4, Dim>(4 * i, 0).transpose().rowwise().reverse());
    }
    return;
  }

  inline void getEnergy(double &energy) const
  {
    energy = 0.0;
    for (int i = 0; i < N; i++) {
      energy += 4.0 * b.row(4 * i + 2).squaredNorm() * T1(i) +
          12.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) * T2(i) +
          12.0 * b.row(4 * i + 3).squaredNorm() * T3(i);
    }
    return;
  }

  inline const Eigen::MatrixX3d &getCoeffs(void) const { return b; }

  inline void calGrads_CT()
  {
    gdT.setZero();
    gdC.setZero();
    // addGradJbyC
    for (int i = 0; i < N; i++) {
      gdC.row(4 * i + 3) =
          12.0 * b.row(4 * i + 2) * T2(i) + 24.0 * b.row(4 * i + 3) * T3(i);
      gdC.row(4 * i + 2) =
          8.0 * b.row(4 * i + 2) * T1(i) + 12.0 * b.row(4 * i + 3) * T2(i);
    }
    // addGradJbyT
    for (int i = 0; i < N; i++) {
      gdT(i) = 4.0 * b.row(4 * i + 2).squaredNorm() +
          24.0 * b.row(4 * i + 2).dot(b.row(4 * i + 3)) * T1(i) +
          36.0 * b.row(4 * i + 3).squaredNorm() * T2(i);
    }
    return;
  }

  inline void calGrads_PT()
  {
    A.solveAdj(gdC);

    for (int i = 0; i < N - 1; i++) {
      gdP.col(i) = gdC.row(4 * i + 3).transpose();
    }

    Eigen::Matrix<double, 4, Dim> B1;
    Eigen::Matrix<double, 2, Dim> B2;
    for (int i = 0; i < N - 1; i++) {
      // negative jerk
      B1.row(0) = -6.0 * b.row(i * 4 + 3);

      // negative velocity
      B1.row(1) = -(b.row(i * 4 + 1) + 2.0 * T1(i) * b.row(i * 4 + 2) +
          3.0 * T2(i) * b.row(i * 4 + 3));
      B1.row(2) = B1.row(1);

      // negative acceleration
      B1.row(3) = -(2.0 * b.row(i * 4 + 2) + 6.0 * T1(i) * b.row(i * 4 + 3));

      gdT(i) += B1.cwiseProduct(gdC.block<4, Dim>(4 * i + 2, 0)).sum();
    }

    // negative velocity
    B2.row(0) = -(b.row(4 * N - 3) + 2.0 * T1(N - 1) * b.row(4 * N - 2) +
        3.0 * T2(N - 1) * b.row(4 * N - 1));

    // negative acceleration
    B2.row(1) = -(2.0 * b.row(4 * N - 2) + 6.0 * T1(N - 1) * b.row(4 * N - 1));

    gdT(N - 1) += B2.cwiseProduct(gdC.block<2, Dim>(4 * N - 2, 0)).sum();

    return;
  }
};

// MINCO for s=3 and non-uniform time
template <int Dim>
class MINCO<Dim, 3, false>
{
 public:
  MINCO() = default;
  ~MINCO() { A.destroy(); }

  int N;
  Eigen::Matrix<double, Dim, 3> headPVA;
  Eigen::Matrix<double, Dim, 3> tailPVA;
  BandedSystem A;
  Eigen::MatrixXd b;

  Eigen::VectorXd T1;
  Eigen::VectorXd T2;
  Eigen::VectorXd T3;
  Eigen::VectorXd T4;
  Eigen::VectorXd T5;

  // for outside use
  Eigen::MatrixXd gdP;
  Eigen::MatrixXd gdC;
  Eigen::VectorXd gdT;

  inline void reset(const Eigen::Matrix<double, Dim, 3> &headState,
                    const Eigen::Matrix<double, Dim, 3> &tailState,
                    const int &pieceNum)
  {
    N = pieceNum;
    headPVA = headState;
    tailPVA = tailState;
    A.create(6 * N, 6, 6);
    b.resize(6 * N, Dim);

    T1.resize(N);
    T2.resize(N);
    T3.resize(N);
    T4.resize(N);
    T5.resize(N);

    gdC.resize(6 * N, Dim);
    gdP.resize(Dim, N - 1);
    gdT.resize(N);
    return;
  }

  inline void generate(const Eigen::Matrix<double, Dim, -1> &inPs,
                       const Eigen::VectorXd &ts)
  {
    T1 = ts;
    T2 = T1.cwiseProduct(T1);
    T3 = T2.cwiseProduct(T1);
    T4 = T2.cwiseProduct(T2);
    T5 = T4.cwiseProduct(T1);

    A.reset();
    b.setZero();

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 2.0;
    b.row(0) = headPVA.col(0).transpose();
    b.row(1) = headPVA.col(1).transpose();
    b.row(2) = headPVA.col(2).transpose();

    for (int i = 0; i < N - 1; i++) {
      A(6 * i + 3, 6 * i + 3) = 6.0;
      A(6 * i + 3, 6 * i + 4) = 24.0 * T1(i);
      A(6 * i + 3, 6 * i + 5) = 60.0 * T2(i);
      A(6 * i + 3, 6 * i + 9) = -6.0;
      A(6 * i + 4, 6 * i + 4) = 24.0;
      A(6 * i + 4, 6 * i + 5) = 120.0 * T1(i);
      A(6 * i + 4, 6 * i + 10) = -24.0;
      A(6 * i + 5, 6 * i) = 1.0;
      A(6 * i + 5, 6 * i + 1) = T1(i);
      A(6 * i + 5, 6 * i + 2) = T2(i);
      A(6 * i + 5, 6 * i + 3) = T3(i);
      A(6 * i + 5, 6 * i + 4) = T4(i);
      A(6 * i + 5, 6 * i + 5) = T5(i);
      A(6 * i + 6, 6 * i) = 1.0;
      A(6 * i + 6, 6 * i + 1) = T1(i);
      A(6 * i + 6, 6 * i + 2) = T2(i);
      A(6 * i + 6, 6 * i + 3) = T3(i);
      A(6 * i + 6, 6 * i + 4) = T4(i);
      A(6 * i + 6, 6 * i + 5) = T5(i);
      A(6 * i + 6, 6 * i + 6) = -1.0;
      A(6 * i + 7, 6 * i + 1) = 1.0;
      A(6 * i + 7, 6 * i + 2) = 2 * T1(i);
      A(6 * i + 7, 6 * i + 3) = 3 * T2(i);
      A(6 * i + 7, 6 * i + 4) = 4 * T3(i);
      A(6 * i + 7, 6 * i + 5) = 5 * T4(i);
      A(6 * i + 7, 6 * i + 7) = -1.0;
      A(6 * i + 8, 6 * i + 2) = 2.0;
      A(6 * i + 8, 6 * i + 3) = 6 * T1(i);
      A(6 * i + 8, 6 * i + 4) = 12 * T2(i);
      A(6 * i + 8, 6 * i + 5) = 20 * T3(i);
      A(6 * i + 8, 6 * i + 8) = -2.0;

      b.row(6 * i + 5) = inPs.col(i).transpose();
    }

    A(6 * N - 3, 6 * N - 6) = 1.0;
    A(6 * N - 3, 6 * N - 5) = T1(N - 1);
    A(6 * N - 3, 6 * N - 4) = T2(N - 1);
    A(6 * N - 3, 6 * N - 3) = T3(N - 1);
    A(6 * N - 3, 6 * N - 2) = T4(N - 1);
    A(6 * N - 3, 6 * N - 1) = T5(N - 1);
    A(6 * N - 2, 6 * N - 5) = 1.0;
    A(6 * N - 2, 6 * N - 4) = 2 * T1(N - 1);
    A(6 * N - 2, 6 * N - 3) = 3 * T2(N - 1);
    A(6 * N - 2, 6 * N - 2) = 4 * T3(N - 1);
    A(6 * N - 2, 6 * N - 1) = 5 * T4(N - 1);
    A(6 * N - 1, 6 * N - 4) = 2;
    A(6 * N - 1, 6 * N - 3) = 6 * T1(N - 1);
    A(6 * N - 1, 6 * N - 2) = 12 * T2(N - 1);
    A(6 * N - 1, 6 * N - 1) = 20 * T3(N - 1);

    b.row(6 * N - 3) = tailPVA.col(0).transpose();
    b.row(6 * N - 2) = tailPVA.col(1).transpose();
    b.row(6 * N - 1) = tailPVA.col(2).transpose();

    A.factorizeLU();
    A.solve(b);

    return;
  }

  inline void getTrajectory(Trajectory<Dim, 5> &traj) const
  {
    traj.clear();
    traj.reserve(N);
    for (int i = 0; i < N; i++) {
      traj.emplace_back(
          T1(i), b.block<6, Dim>(6 * i, 0).transpose().rowwise().reverse());
    }
    return;
  }

  inline void getEnergy(double &energy) const
  {
    energy = 0.0;
    for (int i = 0; i < N; i++) {
      energy += 36.0 * b.row(6 * i + 3).squaredNorm() * T1(i) +
          144.0 * b.row(6 * i + 4).dot(b.row(6 * i + 3)) * T2(i) +
          192.0 * b.row(6 * i + 4).squaredNorm() * T3(i) +
          240.0 * b.row(6 * i + 5).dot(b.row(6 * i + 3)) * T3(i) +
          720.0 * b.row(6 * i + 5).dot(b.row(6 * i + 4)) * T4(i) +
          720.0 * b.row(6 * i + 5).squaredNorm() * T5(i);
    }
    return;
  }

  inline const Eigen::MatrixX3d &getCoeffs(void) const { return b; }

  inline void calGrads_CT()
  {
    gdT.setZero();
    gdC.setZero();
    // addGradJbyC
    for (int i = 0; i < N; i++) {
      gdC.row(6 * i + 5) += 240.0 * b.row(6 * i + 3) * T3(i) +
          720.0 * b.row(6 * i + 4) * T4(i) +
          1440.0 * b.row(6 * i + 5) * T5(i);
      gdC.row(6 * i + 4) += 144.0 * b.row(6 * i + 3) * T2(i) +
          384.0 * b.row(6 * i + 4) * T3(i) +
          720.0 * b.row(6 * i + 5) * T4(i);
      gdC.row(6 * i + 3) += 72.0 * b.row(6 * i + 3) * T1(i) +
          144.0 * b.row(6 * i + 4) * T2(i) +
          240.0 * b.row(6 * i + 5) * T3(i);
    }
    // addGradJbyT
    for (int i = 0; i < N; i++) {
      gdT(i) += 36.0 * b.row(6 * i + 3).squaredNorm() +
          288.0 * b.row(6 * i + 4).dot(b.row(6 * i + 3)) * T1(i) +
          576.0 * b.row(6 * i + 4).squaredNorm() * T2(i) +
          720.0 * b.row(6 * i + 5).dot(b.row(6 * i + 3)) * T2(i) +
          2880.0 * b.row(6 * i + 5).dot(b.row(6 * i + 4)) * T3(i) +
          3600.0 * b.row(6 * i + 5).squaredNorm() * T4(i);
    }
    return;
  }

  inline void calGrads_PT()
  {
    A.solveAdj(gdC);
    gdP.setZero();
    // addPropCtoP
    for (int i = 0; i < N - 1; i++) {
      gdP.col(i) += gdC.row(6 * i + 5).transpose();
    }
    // addPropCtoT
    Eigen::MatrixXd B1(6, Dim), B2(3, Dim);

    for (int i = 0; i < N - 1; i++) {
      // negative velocity
      B1.row(2) =
          -(b.row(i * 6 + 1) + 2.0 * T1(i) * b.row(i * 6 + 2) +
              3.0 * T2(i) * b.row(i * 6 + 3) + 4.0 * T3(i) * b.row(i * 6 + 4) +
              5.0 * T4(i) * b.row(i * 6 + 5));
      B1.row(3) = B1.row(2);

      // negative acceleration
      B1.row(4) =
          -(2.0 * b.row(i * 6 + 2) + 6.0 * T1(i) * b.row(i * 6 + 3) +
              12.0 * T2(i) * b.row(i * 6 + 4) + 20.0 * T3(i) * b.row(i * 6 + 5));

      // negative jerk
      B1.row(5) = -(6.0 * b.row(i * 6 + 3) + 24.0 * T1(i) * b.row(i * 6 + 4) +
          60.0 * T2(i) * b.row(i * 6 + 5));

      // negative snap
      B1.row(0) = -(24.0 * b.row(i * 6 + 4) + 120.0 * T1(i) * b.row(i * 6 + 5));

      // negative crackle
      B1.row(1) = -120.0 * b.row(i * 6 + 5);

      // negSnp, negCrk, negVel, negVel, negAcc, negJer;

      gdT(i) += B1.cwiseProduct(gdC.block<6, Dim>(6 * i + 3, 0)).sum();
    }
    // negative velocity
    B2.row(0) = -(b.row(6 * N - 5) + 2.0 * T1(N - 1) * b.row(6 * N - 4) +
        3.0 * T2(N - 1) * b.row(6 * N - 3) +
        4.0 * T3(N - 1) * b.row(6 * N - 2) +
        5.0 * T4(N - 1) * b.row(6 * N - 1));

    // negative acceleration
    B2.row(1) = -(2.0 * b.row(6 * N - 4) + 6.0 * T1(N - 1) * b.row(6 * N - 3) +
        12.0 * T2(N - 1) * b.row(6 * N - 2) +
        20.0 * T3(N - 1) * b.row(6 * N - 1));

    // negative jerk
    B2.row(2) = -(6.0 * b.row(6 * N - 3) + 24.0 * T1(N - 1) * b.row(6 * N - 2) +
        60.0 * T2(N - 1) * b.row(6 * N - 1));

    gdT(N - 1) += B2.cwiseProduct(gdC.block<3, Dim>(6 * N - 3, 0)).sum();

    return;
  }
};

// MINCO for s=3 and uniform time
template <int Dim>
class MINCO<Dim, 3, true>
{
 public:
  MINCO() = default;
  ~MINCO() { A.destroy(); }

 public:
  int N;
  Eigen::Matrix<double, Dim, 3> headPVA;
  Eigen::Matrix<double, Dim, 3> tailPVA;
  BandedSystem A;
  Eigen::MatrixXd b;
  Eigen::MatrixXd c, adjScaledGrad;

  Eigen::Matrix<double, 6, 1> t, tInv;

  // for outside use
  Eigen::MatrixXd gdHead;
  Eigen::MatrixXd gdTail;
  Eigen::MatrixXd gdP;
  Eigen::MatrixXd gdC;
  double gdT;

  inline void reset(const Eigen::Matrix<double, Dim, 3> &headState,
                    const Eigen::Matrix<double, Dim, 3> &tailState,
                    const int &pieceNum)
  {
    N = pieceNum;
    headPVA = headState;
    tailPVA = tailState;
    A.create(6 * N, 6, 6);
    b.resize(6 * N, Dim);
    c.resize(6 * N, Dim);
    adjScaledGrad.resize(6 * N, Dim);
    gdC.resize(6 * N, Dim);
    gdP.resize(Dim, N - 1);

    gdHead.resize(Dim, 3);
    gdTail.resize(Dim, 3);

    t(0) = 1.0;

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 2.0;
    for (int i = 0; i < N - 1; i++) {
      A(6 * i + 3, 6 * i + 3) = 6.0;
      A(6 * i + 3, 6 * i + 4) = 24.0;
      A(6 * i + 3, 6 * i + 5) = 60.0;
      A(6 * i + 3, 6 * i + 9) = -6.0;
      A(6 * i + 4, 6 * i + 4) = 24.0;
      A(6 * i + 4, 6 * i + 5) = 120.0;
      A(6 * i + 4, 6 * i + 10) = -24.0;
      A(6 * i + 5, 6 * i) = 1.0;
      A(6 * i + 5, 6 * i + 1) = 1.0;
      A(6 * i + 5, 6 * i + 2) = 1.0;
      A(6 * i + 5, 6 * i + 3) = 1.0;
      A(6 * i + 5, 6 * i + 4) = 1.0;
      A(6 * i + 5, 6 * i + 5) = 1.0;
      A(6 * i + 6, 6 * i) = 1.0;
      A(6 * i + 6, 6 * i + 1) = 1.0;
      A(6 * i + 6, 6 * i + 2) = 1.0;
      A(6 * i + 6, 6 * i + 3) = 1.0;
      A(6 * i + 6, 6 * i + 4) = 1.0;
      A(6 * i + 6, 6 * i + 5) = 1.0;
      A(6 * i + 6, 6 * i + 6) = -1.0;
      A(6 * i + 7, 6 * i + 1) = 1.0;
      A(6 * i + 7, 6 * i + 2) = 2.0;
      A(6 * i + 7, 6 * i + 3) = 3.0;
      A(6 * i + 7, 6 * i + 4) = 4.0;
      A(6 * i + 7, 6 * i + 5) = 5.0;
      A(6 * i + 7, 6 * i + 7) = -1.0;
      A(6 * i + 8, 6 * i + 2) = 2.0;
      A(6 * i + 8, 6 * i + 3) = 6.0;
      A(6 * i + 8, 6 * i + 4) = 12.0;
      A(6 * i + 8, 6 * i + 5) = 20.0;
      A(6 * i + 8, 6 * i + 8) = -2.0;
    }
    A(6 * N - 3, 6 * N - 6) = 1.0;
    A(6 * N - 3, 6 * N - 5) = 1.0;
    A(6 * N - 3, 6 * N - 4) = 1.0;
    A(6 * N - 3, 6 * N - 3) = 1.0;
    A(6 * N - 3, 6 * N - 2) = 1.0;
    A(6 * N - 3, 6 * N - 1) = 1.0;
    A(6 * N - 2, 6 * N - 5) = 1.0;
    A(6 * N - 2, 6 * N - 4) = 2.0;
    A(6 * N - 2, 6 * N - 3) = 3.0;
    A(6 * N - 2, 6 * N - 2) = 4.0;
    A(6 * N - 2, 6 * N - 1) = 5.0;
    A(6 * N - 1, 6 * N - 4) = 2.0;
    A(6 * N - 1, 6 * N - 3) = 6.0;
    A(6 * N - 1, 6 * N - 2) = 12.0;
    A(6 * N - 1, 6 * N - 1) = 20.0;
    A.factorizeLU();

    return;
  }

  inline void generate(const Eigen::Matrix<double, Dim, -1> &inPs,
                       const double &dT)
  {
    t(1) = dT;
    t(2) = t(1) * t(1);
    t(3) = t(2) * t(1);
    t(4) = t(2) * t(2);
    t(5) = t(4) * t(1);
    tInv = t.cwiseInverse();

    b.setZero();
    b.row(0) = headPVA.col(0).transpose();
    b.row(1) = headPVA.col(1).transpose() * t(1);
    b.row(2) = headPVA.col(2).transpose() * t(2);

    for (int i = 0; i < N - 1; i++) {
      b.row(6 * i + 5) = inPs.col(i).transpose();
    }
    b.row(6 * N - 3) = tailPVA.col(0).transpose();
    b.row(6 * N - 2) = tailPVA.col(1).transpose() * t(1);
    b.row(6 * N - 1) = tailPVA.col(2).transpose() * t(2);

    A.solve(b);

    for (int i = 0; i < N; i++) {
      c.block<6, Dim>(6 * i, 0) =
          b.block<6, Dim>(6 * i, 0).array().colwise() * tInv.array();
    }

    return;
  }

  inline void getTrajectory(Trajectory<Dim, 5> &traj) const
  {
    traj.clear();
    traj.reserve(N);
    for (int i = 0; i < N; i++) {
      traj.emplace_back(
          t(1), c.block<6, Dim>(6 * i, 0).transpose().rowwise().reverse());
    }
    return;
  }

  inline void getEnergy(double &energy) const
  {
    energy = 0.0;
    for (int i = 0; i < N; i++) {
      energy += 36.0 * c.row(6 * i + 3).squaredNorm() * t(1) +
                144.0 * c.row(6 * i + 4).dot(c.row(6 * i + 3)) * t(2) +
                192.0 * c.row(6 * i + 4).squaredNorm() * t(3) +
                240.0 * c.row(6 * i + 5).dot(c.row(6 * i + 3)) * t(3) +
                720.0 * c.row(6 * i + 5).dot(c.row(6 * i + 4)) * t(4) +
                720.0 * c.row(6 * i + 5).squaredNorm() * t(5);
    }
    return;
  }

  inline const Eigen::MatrixX3d &getCoeffs(void) const { return c; }

  inline void calGrads_CT()
  {
    gdT = 0.0;
    gdC.setZero();
    // addGradJbyC
    for (int i = 0; i < N; i++) {
      gdC.row(6 * i + 5) = 240.0 * c.row(6 * i + 3) * t(3) +
          720.0 * c.row(6 * i + 4) * t(4) +
          1440.0 * c.row(6 * i + 5) * t(5);
      gdC.row(6 * i + 4) = 144.0 * c.row(6 * i + 3) * t(2) +
          384.0 * c.row(6 * i + 4) * t(3) +
          720.0 * c.row(6 * i + 5) * t(4);
      gdC.row(6 * i + 3) = 72.0 * c.row(6 * i + 3) * t(1) +
          144.0 * c.row(6 * i + 4) * t(2) +
          240.0 * c.row(6 * i + 5) * t(3);
//      gdC.block<3, Dim>(6 * i, 0).setZero();
    }
    // addGradJbyT
    for (int i = 0; i < N; i++) {
      gdT += 36.0 * c.row(6 * i + 3).squaredNorm() +
          288.0 * c.row(6 * i + 4).dot(c.row(6 * i + 3)) * t(1) +
          576.0 * c.row(6 * i + 4).squaredNorm() * t(2) +
          720.0 * c.row(6 * i + 5).dot(c.row(6 * i + 3)) * t(2) +
          2880.0 * c.row(6 * i + 5).dot(c.row(6 * i + 4)) * t(3) +
          3600.0 * c.row(6 * i + 5).squaredNorm() * t(4);
    }
    return;
  }

  inline void calGrads_PT()
  {
    for (int i = 0; i < N; i++) {
      adjScaledGrad.block<6, Dim>(6 * i, 0) =
          gdC.block<6, Dim>(6 * i, 0).array().colwise() * tInv.array();
    }
    A.solveAdj(adjScaledGrad);

    for (int i = 0; i < N - 1; i++) {
      gdP.col(i) = adjScaledGrad.row(6 * i + 5).transpose();
    }
    gdHead = adjScaledGrad.topRows(3).transpose() * t.head<3>().asDiagonal();
    gdTail = adjScaledGrad.bottomRows(3).transpose() * t.head<3>().asDiagonal();

    gdT += headPVA.col(1).dot(adjScaledGrad.row(1));
    gdT += headPVA.col(2).dot(adjScaledGrad.row(2)) * 2.0 * t(1);
    gdT += tailPVA.col(1).dot(adjScaledGrad.row(6 * N - 2));
    gdT += tailPVA.col(2).dot(adjScaledGrad.row(6 * N - 1)) * 2.0 * t(1);
    Eigen::Matrix<double, 6, 1> gdtInv;
    gdtInv(0) = 0.0;
    gdtInv(1) = -1.0 * tInv(2);
    gdtInv(2) = -2.0 * tInv(3);
    gdtInv(3) = -3.0 * tInv(4);
    gdtInv(4) = -4.0 * tInv(5);
    gdtInv(5) = -5.0 * tInv(5) * tInv(1);
    const Eigen::VectorXd gdcol = gdC.cwiseProduct(b).rowwise().sum();
    for (int i = 0; i < N; i++) {
      gdT += gdtInv.dot(gdcol.segment<6>(6 * i));
    }
    return;
  }
};

// MINCO for s=4 and non-uniform time
template <int Dim>
class MINCO<Dim, 4, false>
{
 public:
  int N;
  Eigen::Matrix<double, Dim, 4> headPVAJ;
  Eigen::Matrix<double, Dim, 4> tailPVAJ;

  BandedSystem A;
  Eigen::MatrixXd b;

  Eigen::VectorXd T1;
  Eigen::VectorXd T2;
  Eigen::VectorXd T3;
  Eigen::VectorXd T4;
  Eigen::VectorXd T5;
  Eigen::VectorXd T6;
  Eigen::VectorXd T7;

  // for outside use
  Eigen::MatrixXd gdHead;
  Eigen::MatrixXd gdTail;
  Eigen::MatrixXd gdP;
  Eigen::MatrixXd gdC;
  Eigen::VectorXd gdT;

  MINCO() = default;
  ~MINCO() { A.destroy(); }

  inline void reset(const Eigen::Matrix<double, Dim, 4> &headState,
                    const Eigen::Matrix<double, Dim, 4> &tailState,
                    const int &pieceNum)
  {
    headPVAJ = headState;
    tailPVAJ = tailState;

    N = pieceNum;
    A.create(8 * N, 8, 8);
    b.resize(8 * N, Dim);

    T1.resize(N);
    T2.resize(N);
    T3.resize(N);
    T4.resize(N);
    T5.resize(N);
    T6.resize(N);
    T7.resize(N);

    gdC.resize(8 * N, Dim);
    gdP.resize(Dim, N - 1);
    gdT.resize(N);

    gdHead.resize(Dim, 4);
    gdTail.resize(Dim, 4);

    return;
  }

  inline void generate(const Eigen::Matrix<double, Dim, -1> &inPs,
                       const Eigen::VectorXd &ts)
  {
    T1 = ts;
    T2 = T1.cwiseProduct(T1);
    T3 = T2.cwiseProduct(T1);
    T4 = T2.cwiseProduct(T2);
    T5 = T4.cwiseProduct(T1);
    T6 = T4.cwiseProduct(T2);
    T7 = T4.cwiseProduct(T3);

    A.reset();
    b.setZero();

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 2.0;
    A(3, 3) = 6.0;
    b.row(0) = headPVAJ.col(0).transpose();
    b.row(1) = headPVAJ.col(1).transpose();
    b.row(2) = headPVAJ.col(2).transpose();
    b.row(3) = headPVAJ.col(3).transpose();

    for (int i = 0; i < N - 1; i++) {
      A(8 * i + 4, 8 * i + 4) = 24.0;
      A(8 * i + 4, 8 * i + 5) = 120.0 * T1(i);
      A(8 * i + 4, 8 * i + 6) = 360.0 * T2(i);
      A(8 * i + 4, 8 * i + 7) = 840.0 * T3(i);
      A(8 * i + 4, 8 * i + 12) = -24.0;
      A(8 * i + 5, 8 * i + 5) = 120.0;
      A(8 * i + 5, 8 * i + 6) = 720.0 * T1(i);
      A(8 * i + 5, 8 * i + 7) = 2520.0 * T2(i);
      A(8 * i + 5, 8 * i + 13) = -120.0;
      A(8 * i + 6, 8 * i + 6) = 720.0;
      A(8 * i + 6, 8 * i + 7) = 5040.0 * T1(i);
      A(8 * i + 6, 8 * i + 14) = -720.0;
      A(8 * i + 7, 8 * i) = 1.0;
      A(8 * i + 7, 8 * i + 1) = T1(i);
      A(8 * i + 7, 8 * i + 2) = T2(i);
      A(8 * i + 7, 8 * i + 3) = T3(i);
      A(8 * i + 7, 8 * i + 4) = T4(i);
      A(8 * i + 7, 8 * i + 5) = T5(i);
      A(8 * i + 7, 8 * i + 6) = T6(i);
      A(8 * i + 7, 8 * i + 7) = T7(i);
      A(8 * i + 8, 8 * i) = 1.0;
      A(8 * i + 8, 8 * i + 1) = T1(i);
      A(8 * i + 8, 8 * i + 2) = T2(i);
      A(8 * i + 8, 8 * i + 3) = T3(i);
      A(8 * i + 8, 8 * i + 4) = T4(i);
      A(8 * i + 8, 8 * i + 5) = T5(i);
      A(8 * i + 8, 8 * i + 6) = T6(i);
      A(8 * i + 8, 8 * i + 7) = T7(i);
      A(8 * i + 8, 8 * i + 8) = -1.0;
      A(8 * i + 9, 8 * i + 1) = 1.0;
      A(8 * i + 9, 8 * i + 2) = 2.0 * T1(i);
      A(8 * i + 9, 8 * i + 3) = 3.0 * T2(i);
      A(8 * i + 9, 8 * i + 4) = 4.0 * T3(i);
      A(8 * i + 9, 8 * i + 5) = 5.0 * T4(i);
      A(8 * i + 9, 8 * i + 6) = 6.0 * T5(i);
      A(8 * i + 9, 8 * i + 7) = 7.0 * T6(i);
      A(8 * i + 9, 8 * i + 9) = -1.0;
      A(8 * i + 10, 8 * i + 2) = 2.0;
      A(8 * i + 10, 8 * i + 3) = 6.0 * T1(i);
      A(8 * i + 10, 8 * i + 4) = 12.0 * T2(i);
      A(8 * i + 10, 8 * i + 5) = 20.0 * T3(i);
      A(8 * i + 10, 8 * i + 6) = 30.0 * T4(i);
      A(8 * i + 10, 8 * i + 7) = 42.0 * T5(i);
      A(8 * i + 10, 8 * i + 10) = -2.0;
      A(8 * i + 11, 8 * i + 3) = 6.0;
      A(8 * i + 11, 8 * i + 4) = 24.0 * T1(i);
      A(8 * i + 11, 8 * i + 5) = 60.0 * T2(i);
      A(8 * i + 11, 8 * i + 6) = 120.0 * T3(i);
      A(8 * i + 11, 8 * i + 7) = 210.0 * T4(i);
      A(8 * i + 11, 8 * i + 11) = -6.0;

      b.row(8 * i + 7) = inPs.col(i).transpose();
    }

    A(8 * N - 4, 8 * N - 8) = 1.0;
    A(8 * N - 4, 8 * N - 7) = T1(N - 1);
    A(8 * N - 4, 8 * N - 6) = T2(N - 1);
    A(8 * N - 4, 8 * N - 5) = T3(N - 1);
    A(8 * N - 4, 8 * N - 4) = T4(N - 1);
    A(8 * N - 4, 8 * N - 3) = T5(N - 1);
    A(8 * N - 4, 8 * N - 2) = T6(N - 1);
    A(8 * N - 4, 8 * N - 1) = T7(N - 1);
    A(8 * N - 3, 8 * N - 7) = 1.0;
    A(8 * N - 3, 8 * N - 6) = 2.0 * T1(N - 1);
    A(8 * N - 3, 8 * N - 5) = 3.0 * T2(N - 1);
    A(8 * N - 3, 8 * N - 4) = 4.0 * T3(N - 1);
    A(8 * N - 3, 8 * N - 3) = 5.0 * T4(N - 1);
    A(8 * N - 3, 8 * N - 2) = 6.0 * T5(N - 1);
    A(8 * N - 3, 8 * N - 1) = 7.0 * T6(N - 1);
    A(8 * N - 2, 8 * N - 6) = 2.0;
    A(8 * N - 2, 8 * N - 5) = 6.0 * T1(N - 1);
    A(8 * N - 2, 8 * N - 4) = 12.0 * T2(N - 1);
    A(8 * N - 2, 8 * N - 3) = 20.0 * T3(N - 1);
    A(8 * N - 2, 8 * N - 2) = 30.0 * T4(N - 1);
    A(8 * N - 2, 8 * N - 1) = 42.0 * T5(N - 1);
    A(8 * N - 1, 8 * N - 5) = 6.0;
    A(8 * N - 1, 8 * N - 4) = 24.0 * T1(N - 1);
    A(8 * N - 1, 8 * N - 3) = 60.0 * T2(N - 1);
    A(8 * N - 1, 8 * N - 2) = 120.0 * T3(N - 1);
    A(8 * N - 1, 8 * N - 1) = 210.0 * T4(N - 1);

    b.row(8 * N - 4) = tailPVAJ.col(0).transpose();
    b.row(8 * N - 3) = tailPVAJ.col(1).transpose();
    b.row(8 * N - 2) = tailPVAJ.col(2).transpose();
    b.row(8 * N - 1) = tailPVAJ.col(3).transpose();

    A.factorizeLU();
    A.solve(b);

    return;
  }

  inline void getTrajectory(Trajectory<Dim, 7> &traj) const
  {
    traj.clear();
    traj.reserve(N);
    for (int i = 0; i < N; i++) {
      traj.emplace_back(
          T1(i), b.block<8, Dim>(8 * i, 0).transpose().rowwise().reverse());
    }
    return;
  }

  inline void getEnergy(double &energy) const
  {
    energy = 0.0;
    for (int i = 0; i < N; i++) {
      energy += 576.0 * b.row(8 * i + 4).squaredNorm() * T1(i) +
          2880.0 * b.row(8 * i + 4).dot(b.row(8 * i + 5)) * T2(i) +
          4800.0 * b.row(8 * i + 5).squaredNorm() * T3(i) +
          5760.0 * b.row(8 * i + 4).dot(b.row(8 * i + 6)) * T3(i) +
          21600.0 * b.row(8 * i + 5).dot(b.row(8 * i + 6)) * T4(i) +
          10080.0 * b.row(8 * i + 4).dot(b.row(8 * i + 7)) * T4(i) +
          25920.0 * b.row(8 * i + 6).squaredNorm() * T5(i) +
          40320.0 * b.row(8 * i + 5).dot(b.row(8 * i + 7)) * T5(i) +
          100800.0 * b.row(8 * i + 6).dot(b.row(8 * i + 7)) * T6(i) +
          100800.0 * b.row(8 * i + 7).squaredNorm() * T7(i);
    }
    return;
  }

  inline const Eigen::MatrixX3d &getCoeffs(void) const { return b; }

  inline void calGrads_CT(){
    gdT.setZero();
    gdC.setZero();
    // addGradJbyC
    for (int i = 0; i < N; i++) {
      gdC.row(8 * i + 7) = 10080.0 * b.row(8 * i + 4) * T4(i) +
          40320.0 * b.row(8 * i + 5) * T5(i) +
          100800.0 * b.row(8 * i + 6) * T6(i) +
          201600.0 * b.row(8 * i + 7) * T7(i);
      gdC.row(8 * i + 6) = 5760.0 * b.row(8 * i + 4) * T3(i) +
          21600.0 * b.row(8 * i + 5) * T4(i) +
          51840.0 * b.row(8 * i + 6) * T5(i) +
          100800.0 * b.row(8 * i + 7) * T6(i);
      gdC.row(8 * i + 5) =
          2880.0 * b.row(8 * i + 4) * T2(i) + 9600.0 * b.row(8 * i + 5) * T3(i) +
              21600.0 * b.row(8 * i + 6) * T4(i) + 40320.0 * b.row(8 * i + 7) * T5(i);
      gdC.row(8 * i + 4) =
          1152.0 * b.row(8 * i + 4) * T1(i) + 2880.0 * b.row(8 * i + 5) * T2(i) +
              5760.0 * b.row(8 * i + 6) * T3(i) + 10080.0 * b.row(8 * i + 7) * T4(i);
//      gdC.block<4, Dim>(8 * i, 0).setZero();
    }

    // addGradJbyT
    for (int i = 0; i < N; i++) {
      gdT(i) += 576.0 * b.row(8 * i + 4).squaredNorm() +
          5760.0 * b.row(8 * i + 4).dot(b.row(8 * i + 5)) * T1(i) +
          14400.0 * b.row(8 * i + 5).squaredNorm() * T2(i) +
          17280.0 * b.row(8 * i + 4).dot(b.row(8 * i + 6)) * T2(i) +
          86400.0 * b.row(8 * i + 5).dot(b.row(8 * i + 6)) * T3(i) +
          40320.0 * b.row(8 * i + 4).dot(b.row(8 * i + 7)) * T3(i) +
          129600.0 * b.row(8 * i + 6).squaredNorm() * T4(i) +
          201600.0 * b.row(8 * i + 5).dot(b.row(8 * i + 7)) * T4(i) +
          604800.0 * b.row(8 * i + 6).dot(b.row(8 * i + 7)) * T5(i) +
          705600.0 * b.row(8 * i + 7).squaredNorm() * T6(i);
    }
    return;
  }

  inline void calGrads_PT()
  {
    A.solveAdj(gdC);

    // addPropCtoP
    for (int i = 0; i < N - 1; i++) {
      gdP.col(i) = gdC.row(8 * i + 7).transpose();
    }
    gdHead = gdC.topRows(4).transpose();
    gdTail = gdC.bottomRows(4).transpose();

    // addPropCtoT
    Eigen::Matrix<double, 8, Dim> B1;
    Eigen::Matrix<double, 4, Dim> B2;
    for (int i = 0; i < N - 1; i++) {
      // negative velocity
      B1.row(3) =
          -(b.row(i * 8 + 1) + 2.0 * T1(i) * b.row(i * 8 + 2) +
              3.0 * T2(i) * b.row(i * 8 + 3) + 4.0 * T3(i) * b.row(i * 8 + 4) +
              5.0 * T4(i) * b.row(i * 8 + 5) + 6.0 * T5(i) * b.row(i * 8 + 6) +
              7.0 * T6(i) * b.row(i * 8 + 7));
      B1.row(4) = B1.row(3);

      // negative acceleration
      B1.row(5) =
          -(2.0 * b.row(i * 8 + 2) + 6.0 * T1(i) * b.row(i * 8 + 3) +
              12.0 * T2(i) * b.row(i * 8 + 4) + 20.0 * T3(i) * b.row(i * 8 + 5) +
              30.0 * T4(i) * b.row(i * 8 + 6) + 42.0 * T5(i) * b.row(i * 8 + 7));

      // negative jerk
      B1.row(6) =
          -(6.0 * b.row(i * 8 + 3) + 24.0 * T1(i) * b.row(i * 8 + 4) +
              60.0 * T2(i) * b.row(i * 8 + 5) + 120.0 * T3(i) * b.row(i * 8 + 6) +
              210.0 * T4(i) * b.row(i * 8 + 7));

      // negative snap
      B1.row(7) =
          -(24.0 * b.row(i * 8 + 4) + 120.0 * T1(i) * b.row(i * 8 + 5) +
              360.0 * T2(i) * b.row(i * 8 + 6) + 840.0 * T3(i) * b.row(i * 8 + 7));

      // negative crackle
      B1.row(0) = -(120.0 * b.row(i * 8 + 5) + 720.0 * T1(i) * b.row(i * 8 + 6) +
          2520.0 * T2(i) * b.row(i * 8 + 7));

      // negative d_crackle
      B1.row(1) = -(720.0 * b.row(i * 8 + 6) + 5040.0 * T1(i) * b.row(i * 8 + 7));

      // negative dd_crackle
      B1.row(2) = -5040.0 * b.row(i * 8 + 7);

      gdT(i) += B1.cwiseProduct(gdC.block<8, Dim>(8 * i + 4, 0)).sum();
    }

    // negative velocity
    B2.row(0) = -(
        b.row(8 * N - 7) + 2.0 * T1(N - 1) * b.row(8 * N - 6) +
            3.0 * T2(N - 1) * b.row(8 * N - 5) + 4.0 * T3(N - 1) * b.row(8 * N - 4) +
            5.0 * T4(N - 1) * b.row(8 * N - 3) + 6.0 * T5(N - 1) * b.row(8 * N - 2) +
            7.0 * T6(N - 1) * b.row(8 * N - 1));

    // negative acceleration
    B2.row(1) = -(2.0 * b.row(8 * N - 6) + 6.0 * T1(N - 1) * b.row(8 * N - 5) +
        12.0 * T2(N - 1) * b.row(8 * N - 4) +
        20.0 * T3(N - 1) * b.row(8 * N - 3) +
        30.0 * T4(N - 1) * b.row(8 * N - 2) +
        42.0 * T5(N - 1) * b.row(8 * N - 1));

    // negative jerk
    B2.row(2) = -(6.0 * b.row(8 * N - 5) + 24.0 * T1(N - 1) * b.row(8 * N - 4) +
        60.0 * T2(N - 1) * b.row(8 * N - 3) +
        120.0 * T3(N - 1) * b.row(8 * N - 2) +
        210.0 * T4(N - 1) * b.row(8 * N - 1));

    // negative snap
    B2.row(3) = -(24.0 * b.row(8 * N - 4) + 120.0 * T1(N - 1) * b.row(8 * N - 3) +
        360.0 * T2(N - 1) * b.row(8 * N - 2) +
        840.0 * T3(N - 1) * b.row(8 * N - 1));

    gdT(N - 1) += B2.cwiseProduct(gdC.block<4, Dim>(8 * N - 4, 0)).sum();
  }

};

// MINCO for s=4 and uniform time
template <int Dim>
class MINCO<Dim, 4, true>{
 public:
  int N;
  Eigen::Matrix<double, Dim, 4> headPVAJ;
  Eigen::Matrix<double, Dim, 4> tailPVAJ;

  BandedSystem A;
  Eigen::MatrixXd b;
  Eigen::MatrixXd c, adjScaledGrad;
  Eigen::Matrix<double, 8, 1> t, tInv;

  // for outside use
  Eigen::MatrixXd gdHead;
  Eigen::MatrixXd gdTail;
  Eigen::MatrixXd gdP;
  Eigen::MatrixXd gdC;
  double gdT;

  MINCO() = default;
  ~MINCO() { A.destroy(); }

  inline void reset(const Eigen::Matrix<double, Dim, 4> &headState,
                    const Eigen::Matrix<double, Dim, 4> &tailState,
                    const int &pieceNum)
  {
    headPVAJ = headState;
    tailPVAJ = tailState;

    N = pieceNum;
    A.create(8 * N, 8, 8);
    b.resize(8 * N, Dim);
    c.resize(8 * N, Dim);
    adjScaledGrad.resize(8 * N, Dim);

    gdC.resize(8 * N, Dim);
    gdP.resize(Dim, N - 1);
    gdHead.resize(Dim, 4);
    gdTail.resize(Dim, 4);

    t(0) = 1.0;

    A(0, 0) = 1.0;
    A(1, 1) = 1.0;
    A(2, 2) = 2.0;
    A(3, 3) = 6.0;
    for (int i = 0; i < N - 1; i++) {
      A(8 * i + 4, 8 * i + 4) = 24.0;
      A(8 * i + 4, 8 * i + 5) = 120.0;
      A(8 * i + 4, 8 * i + 6) = 360.0;
      A(8 * i + 4, 8 * i + 7) = 840.0;
      A(8 * i + 4, 8 * i + 12) = -24.0;

      A(8 * i + 5, 8 * i + 5) = 120.0;
      A(8 * i + 5, 8 * i + 6) = 720.0;
      A(8 * i + 5, 8 * i + 7) = 2520.0;
      A(8 * i + 5, 8 * i + 13) = -120.0;

      A(8 * i + 6, 8 * i + 6) = 720.0;
      A(8 * i + 6, 8 * i + 7) = 5040.0;
      A(8 * i + 6, 8 * i + 14) = -720.0;

      A(8 * i + 7, 8 * i) = 1.0;
      A(8 * i + 7, 8 * i + 1) = 1.0;
      A(8 * i + 7, 8 * i + 2) = 1.0;
      A(8 * i + 7, 8 * i + 3) = 1.0;
      A(8 * i + 7, 8 * i + 4) = 1.0;
      A(8 * i + 7, 8 * i + 5) = 1.0;
      A(8 * i + 7, 8 * i + 6) = 1.0;
      A(8 * i + 7, 8 * i + 7) = 1.0;

      A(8 * i + 8, 8 * i) = 1.0;
      A(8 * i + 8, 8 * i + 1) = 1.0;
      A(8 * i + 8, 8 * i + 2) = 1.0;
      A(8 * i + 8, 8 * i + 3) = 1.0;
      A(8 * i + 8, 8 * i + 4) = 1.0;
      A(8 * i + 8, 8 * i + 5) = 1.0;
      A(8 * i + 8, 8 * i + 6) = 1.0;
      A(8 * i + 8, 8 * i + 7) = 1.0;
      A(8 * i + 8, 8 * i + 8) = -1.0;

      A(8 * i + 9, 8 * i + 1) = 1.0;
      A(8 * i + 9, 8 * i + 2) = 2.0;
      A(8 * i + 9, 8 * i + 3) = 3.0;
      A(8 * i + 9, 8 * i + 4) = 4.0;
      A(8 * i + 9, 8 * i + 5) = 5.0;
      A(8 * i + 9, 8 * i + 6) = 6.0;
      A(8 * i + 9, 8 * i + 7) = 7.0;
      A(8 * i + 9, 8 * i + 9) = -1.0;

      A(8 * i + 10, 8 * i + 2) = 2.0;
      A(8 * i + 10, 8 * i + 3) = 6.0;
      A(8 * i + 10, 8 * i + 4) = 12.0;
      A(8 * i + 10, 8 * i + 5) = 20.0;
      A(8 * i + 10, 8 * i + 6) = 30.0;
      A(8 * i + 10, 8 * i + 7) = 42.0;
      A(8 * i + 10, 8 * i + 10) = -2.0;

      A(8 * i + 11, 8 * i + 3) = 6.0;
      A(8 * i + 11, 8 * i + 4) = 24.0;
      A(8 * i + 11, 8 * i + 5) = 60.0;
      A(8 * i + 11, 8 * i + 6) = 120.0;
      A(8 * i + 11, 8 * i + 7) = 210.0;
      A(8 * i + 11, 8 * i + 11) = -6.0;
    }
    A(8 * N - 4, 8 * N - 8) = 1.0;
    A(8 * N - 4, 8 * N - 7) = 1.0;
    A(8 * N - 4, 8 * N - 6) = 1.0;
    A(8 * N - 4, 8 * N - 5) = 1.0;
    A(8 * N - 4, 8 * N - 4) = 1.0;
    A(8 * N - 4, 8 * N - 3) = 1.0;
    A(8 * N - 4, 8 * N - 2) = 1.0;
    A(8 * N - 4, 8 * N - 1) = 1.0;

    A(8 * N - 3, 8 * N - 7) = 1.0;
    A(8 * N - 3, 8 * N - 6) = 2.0;
    A(8 * N - 3, 8 * N - 5) = 3.0;
    A(8 * N - 3, 8 * N - 4) = 4.0;
    A(8 * N - 3, 8 * N - 3) = 5.0;
    A(8 * N - 3, 8 * N - 2) = 6.0;
    A(8 * N - 3, 8 * N - 1) = 7.0;

    A(8 * N - 2, 8 * N - 6) = 2.0;
    A(8 * N - 2, 8 * N - 5) = 6.0;
    A(8 * N - 2, 8 * N - 4) = 12.0;
    A(8 * N - 2, 8 * N - 3) = 20.0;
    A(8 * N - 2, 8 * N - 2) = 30.0;
    A(8 * N - 2, 8 * N - 1) = 42.0;

    A(8 * N - 1, 8 * N - 5) = 6.0;
    A(8 * N - 1, 8 * N - 4) = 24.0;
    A(8 * N - 1, 8 * N - 3) = 60.0;
    A(8 * N - 1, 8 * N - 2) = 120.0;
    A(8 * N - 1, 8 * N - 1) = 210.0;

    A.factorizeLU();

    return;
  }

  inline void generate(const Eigen::Matrix<double, Dim, -1> &inPs,
                       const double &dT)
  {
    t(1) = dT;
    t(2) = t(1) * t(1);
    t(3) = t(2) * t(1);
    t(4) = t(2) * t(2);
    t(5) = t(3) * t(2);
    t(6) = t(3) * t(3);
    t(7) = t(4) * t(3);
    tInv = t.cwiseInverse();

    b.setZero();
    b.row(0) = headPVAJ.col(0).transpose();
    b.row(1) = headPVAJ.col(1).transpose() * t(1);
    b.row(2) = headPVAJ.col(2).transpose() * t(2);
    b.row(3) = headPVAJ.col(3).transpose() * t(3);

    for (int i = 0; i < N - 1; i++) {
      b.row(8 * i + 7) = inPs.col(i).transpose();
    }
    b.row(8 * N - 4) = tailPVAJ.col(0).transpose();
    b.row(8 * N - 3) = tailPVAJ.col(1).transpose() * t(1);
    b.row(8 * N - 2) = tailPVAJ.col(2).transpose() * t(2);
    b.row(8 * N - 1) = tailPVAJ.col(3).transpose() * t(3);

    A.solve(b);

    for (int i = 0; i < N; i++) {
      c.block<8, Dim>(8 * i, 0) =
          b.block<8, Dim>(8 * i, 0).array().colwise() * tInv.array();
    }
    return;
  }

  inline void getTrajectory(Trajectory<Dim, 7> &traj) const
  {
    traj.clear();
    traj.reserve(N);
    for (int i = 0; i < N; i++) {
      traj.emplace_back(
          t(1), c.block<8, Dim>(8 * i, 0).transpose().rowwise().reverse());
    }
    return;
  }

  inline void getEnergy(double &energy) const
  {
    energy = 0.0;
    for (int i = 0; i < N; i++) {
      energy += 576.0 * c.row(8 * i + 4).squaredNorm() * t(1) +
                2880.0 * c.row(8 * i + 4).dot(c.row(8 * i + 5)) * t(2) +
                4800.0 * c.row(8 * i + 5).squaredNorm() * t(3) +
                5760.0 * c.row(8 * i + 4).dot(c.row(8 * i + 6)) * t(3) +
                21600.0 * c.row(8 * i + 5).dot(c.row(8 * i + 6)) * t(4) +
                10080.0 * c.row(8 * i + 4).dot(c.row(8 * i + 7)) * t(4) +
                25920.0 * c.row(8 * i + 6).squaredNorm() * t(5) +
                40320.0 * c.row(8 * i + 5).dot(c.row(8 * i + 7)) * t(5) +
                100800.0 * c.row(8 * i + 6).dot(c.row(8 * i + 7)) * t(6) +
                100800.0 * c.row(8 * i + 7).squaredNorm() * t(7);
    }
    return;
  }

  inline const Eigen::MatrixX3d &getCoeffs(void) const { return c; }

  inline void calGrads_CT(){
    gdT = 0.0;
    gdC.setZero();
    // addGradJbyC
    for (int i = 0; i < N; i++) {
      gdC.row(8 * i + 7) = 10080.0 * c.row(8 * i + 4) * t(4) +
                           40320.0 * c.row(8 * i + 5) * t(5) +
                           100800.0 * c.row(8 * i + 6) * t(6) +
                           201600.0 * c.row(8 * i + 7) * t(7);
      gdC.row(8 * i + 6) = 5760.0 * c.row(8 * i + 4) * t(3) +
                           21600.0 * c.row(8 * i + 5) * t(4) +
                           51840.0 * c.row(8 * i + 6) * t(5) +
                           100800.0 * c.row(8 * i + 7) * t(6);
      gdC.row(8 * i + 5) = 2880.0 * c.row(8 * i + 4) * t(2) +
                           9600.0 * c.row(8 * i + 5) * t(3) +
                           21600.0 * c.row(8 * i + 6) * t(4) +
                           40320.0 * c.row(8 * i + 7) * t(5);
      gdC.row(8 * i + 4) = 1152.0 * c.row(8 * i + 4) * t(1) +
                           2880.0 * c.row(8 * i + 5) * t(2) +
                           5760.0 * c.row(8 * i + 6) * t(3) +
                           10080.0 * c.row(8 * i + 7) * t(4);
//      gdC.block<4, Dim>(8 * i, 0).setZero();
    }

    // addGradJbyT
    for (int i = 0; i < N; i++) {
      gdT += 576.0 * c.row(8 * i + 4).squaredNorm() +
             5760.0 * c.row(8 * i + 4).dot(c.row(8 * i + 5)) * t(1) +
             14400.0 * c.row(8 * i + 5).squaredNorm() * t(2) +
             17280.0 * c.row(8 * i + 4).dot(c.row(8 * i + 6)) * t(2) +
             86400.0 * c.row(8 * i + 5).dot(c.row(8 * i + 6)) * t(3) +
             40320.0 * c.row(8 * i + 4).dot(c.row(8 * i + 7)) * t(3) +
             129600.0 * c.row(8 * i + 6).squaredNorm() * t(4) +
             201600.0 * c.row(8 * i + 5).dot(c.row(8 * i + 7)) * t(4) +
             604800.0 * c.row(8 * i + 6).dot(c.row(8 * i + 7)) * t(5) +
             705600.0 * c.row(8 * i + 7).squaredNorm() * t(6);
    }
    return;
  }

  inline void calGrads_PT()
  {
    for (int i = 0; i < N; i++) {
      adjScaledGrad.block<8, Dim>(8 * i, 0) =
          gdC.block<8, Dim>(8 * i, 0).array().colwise() * tInv.array();
    }
    A.solveAdj(adjScaledGrad);

    // addPropCtoP
    for (int i = 0; i < N - 1; i++) {
      gdP.col(i) = adjScaledGrad.row(8 * i + 7).transpose();
    }
    gdHead = adjScaledGrad.topRows(4).transpose() * t.head<4>().asDiagonal();
    gdTail = adjScaledGrad.bottomRows(4).transpose() * t.head<4>().asDiagonal();

    // addPropCtoT
    gdT += headPVAJ.col(1).dot(adjScaledGrad.row(1));
    gdT += headPVAJ.col(2).dot(adjScaledGrad.row(2)) * 2.0 * t(1);
    gdT += headPVAJ.col(3).dot(adjScaledGrad.row(3)) * 3.0 * t(2);
    gdT += tailPVAJ.col(1).dot(adjScaledGrad.row(8 * N - 3));
    gdT += tailPVAJ.col(2).dot(adjScaledGrad.row(8 * N - 2)) * 2.0 * t(1);
    gdT += tailPVAJ.col(3).dot(adjScaledGrad.row(8 * N - 1)) * 3.0 * t(2);

    Eigen::Matrix<double, 8, 1> gdtInv;
    gdtInv(0) = 0.0;
    gdtInv(1) = -1.0 * tInv(2);
    gdtInv(2) = -2.0 * tInv(3);
    gdtInv(3) = -3.0 * tInv(4);
    gdtInv(4) = -4.0 * tInv(5);
    gdtInv(5) = -5.0 * tInv(6);
    gdtInv(6) = -6.0 * tInv(7);
    gdtInv(7) = -7.0 * tInv(7) * tInv(1);
    const Eigen::VectorXd gdcol = gdC.cwiseProduct(b).rowwise().sum();
    for (int i = 0; i < N; i++) {
      gdT += gdtInv.dot(gdcol.segment<8>(8 * i));
    }
  }

};


}
#endif
