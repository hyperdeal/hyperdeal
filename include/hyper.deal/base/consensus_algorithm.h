// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the hyper.deal authors
//
// This file is part of the hyper.deal library.
//
// The hyper.deal library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hyper.deal.
//
// ---------------------------------------------------------------------

#ifndef DEALII_BASE_CONSENSUS_ALGORITHM
#define DEALII_BASE_CONSENSUS_ALGORITHM

#include <hyper.deal/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>

DEAL_II_NAMESPACE_OPEN

namespace Utilities
{
  namespace MPI
  {
    /**
     * Anonymous version of the Utilities::MPI::ConsensusAlgorithmProcess.
     * Advantage of this class is that users do not have to write their
     * own implementation but can register directly lambdas.
     *
     * @note This will be part of deal.II soon.
     */
    template <typename T1, typename T2>
    class AnonymousConsensusAlgorithmProcess
      : public Utilities::MPI::ConsensusAlgorithmProcess<T1, T2>
    {
    public:
      AnonymousConsensusAlgorithmProcess(
        const std::function<std::vector<unsigned int>()> function1,
        const std::function<void(const unsigned int, std::vector<T1> &)>
          function2 = [](const auto, auto &) {},
        const std::function<
          void(const unsigned int, const std::vector<T1> &, std::vector<T2> &)>
          function3 = [](const auto, const auto &, auto &) {},
        const std::function<void(const unsigned int, std::vector<T2> &)>
          function4 = [](const auto, auto &) {},
        const std::function<void(const unsigned int, const std::vector<T2> &)>
          function5 = [](const auto, const auto &) {})
        : function1(function1)
        , function2(function2)
        , function3(function3)
        , function4(function4)
        , function5(function5)
      {}

      std::vector<unsigned int>
      compute_targets()
      {
        return function1();
      }

      void
      create_request(const unsigned int other_rank,
                     std::vector<T1> &  send_buffer) override
      {
        function2(other_rank, send_buffer);
      }

      void
      answer_request(const unsigned int     other_rank,
                     const std::vector<T1> &buffer_recv,
                     std::vector<T2> &      request_buffer) override
      {
        function3(other_rank, buffer_recv, request_buffer);
      }

      void
      prepare_buffer_for_answer(const unsigned int other_rank,
                                std::vector<T2> &  recv_buffer) override
      {
        function4(other_rank, recv_buffer);
      }

      void
      read_answer(const unsigned int     other_rank,
                  const std::vector<T2> &recv_buffer) override
      {
        function5(other_rank, recv_buffer);
      }

    private:
      const std::function<std::vector<unsigned int>()>        function1;
      const std::function<void(const int, std::vector<T1> &)> function2;
      const std::function<
        void(const unsigned int, const std::vector<T1> &, std::vector<T2> &)>
                                                                    function3;
      const std::function<void(const int, std::vector<T2> &)>       function4;
      const std::function<void(const int, const std::vector<T2> &)> function5;
    };


  } // namespace MPI
} // namespace Utilities

DEAL_II_NAMESPACE_CLOSE

#endif