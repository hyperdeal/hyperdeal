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

#ifndef HYPERDEAL_TIMERS
#define HYPERDEAL_TIMERS

#include <hyper.deal/base/config.h>

#include <hyper.deal/base/memory_consumption.h>

#include <chrono>
#include <ios>
#include <map>
#include <fstream>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

namespace hyperdeal
{
  class Timer
  {
  public:
    void
    reserve(const unsigned int max_size)
    {
      times.reserve(max_size);
    }
      
    void
    reset()
    {
      counter          = 0;
      accumulated_time = 0.0;
    }
    void
    start()
    {
      temp = std::chrono::system_clock::now();
    }
    void
    stop()
    {
      const double dt = std::chrono::duration_cast<std::chrono::microseconds>(
                          std::chrono::system_clock::now() - temp)
                          .count();
        
      accumulated_time += dt;
      
      if(times.capacity() > 0)
          times.push_back(dt);
      
      counter++;
    }

    unsigned int
    get_counter() const
    {
      return counter;
    }

    double
    get_accumulated_time() const
    {
      return accumulated_time;
    }
    
    const std::vector<double> &
    get_log() const
    {
        return times;
    }

  private:
    unsigned int                                       counter = 0;
    std::chrono::time_point<std::chrono::system_clock> temp;
    double                                             accumulated_time = 0.0;
    std::vector<double> times;
  };

  class Timers
  {
    static const unsigned int max_levels = 10;
    static const unsigned int max_timers = 100;
    static const unsigned int max_iterations = 1000;

  public:
    Timers(const bool log_all_calls) : log_all_calls(log_all_calls)
    {
      AssertThrow(!log_all_calls,
                  dealii::StandardExceptions::ExcNotImplemented());

      path.reserve(max_levels);
      timers.reserve(max_timers);

      path.emplace_back("");
    }

    Timer &operator[](const std::string &label)
    {
      const std::string label_ = path.back() + label;

      const auto ptr = map.find(label_);

      if (ptr == map.end())
        {
          timers.resize(timers.size() + 1);
          map[label_] = timers.size() - 1;
          
          if(this->log_all_calls)
            timers.back().reserve(max_iterations);
          return timers.back();
        }
      else
        return timers[ptr->second];
    }

    void
    reset()
    {
      for (auto &timer : timers)
        timer.reset();
    }

    void
    enter(const std::string &label)
    {
      path.emplace_back(path.back() + label + ":");
    }

    void
    leave()
    {
      path.resize(path.size() - 1);
    }

    template <typename StreamType>
    void
    print(const MPI_Comm &comm, StreamType &stream) const
    {
      std::vector<std::pair<std::string, std::array<double, 1>>> list;
      std::vector<std::pair<std::string, unsigned int>>          list_count;

      unsigned int counter     = 0;
      unsigned int max_counter = 0;

      for (const auto &time : map)
        {
          list.emplace_back(time.first,
                            std::array<double, 1>{
                              {timers[time.second].get_accumulated_time() /
                               1000000}});
          list_count.emplace_back(time.first,
                                  timers[time.second].get_counter());

          if (time.first == "id_total")
            max_counter = counter;

          counter++;
        }

      internal::print_(
        stream, comm, list, list_count, {"Time [sec]"}, max_counter);
    }
    
    void
    print_log(const MPI_Comm &comm_global, const std::string & prefix) const
    {
        
      const auto print_statistics = [&](const auto & v, std::string slabel){

          const auto my_rank = dealii::Utilities::MPI::this_mpi_process(comm_global);

          if(my_rank == 0 )
          {
            std::ofstream myfile;
            myfile.open (prefix + "_" + slabel + ".stat");

              for(unsigned int i = 0; i < dealii::Utilities::MPI::n_mpi_processes(comm_global); i++)
              {
                  std::vector<double> recv_data;
                  if(i==0)
                  {
                      recv_data = v;
                  }
                  else
                  {
                  // wait for any request
                  MPI_Status status;
                  auto       ierr = MPI_Probe(MPI_ANY_SOURCE, 110, comm_global, &status);
                  AssertThrowMPI(ierr);

                  // determine number of ghost faces * 2 (since we are considering
                  // pairs)
                  int          len;
                  double dummy;
                  MPI_Get_count(&status,
                                dealii::Utilities::MPI::internal::mpi_type_id(&dummy),
                                &len);

                  recv_data.resize(len);

                  // receive data
                  MPI_Recv(recv_data.data(),
                           len,
                           dealii::Utilities::MPI::internal::mpi_type_id(&dummy),
                           status.MPI_SOURCE,
                           status.MPI_TAG,
                           comm_global,
                           &status);
                  }

                  for(const auto j : recv_data)
                      myfile << i << " " <<  j << std::endl;
              }

              myfile.close();
          }
          else
          {
              unsigned int dummy;
              MPI_Send(v.data(), v.size(), dealii::Utilities::MPI::internal::mpi_type_id(&dummy), 0, 110, comm_global);
          }

          MPI_Barrier(MPI_COMM_WORLD);
    };


      for (const auto &time : map)
        print_statistics(timers[time.second].get_log(), std::string(time.first) );
    }

  private:
    // translator label -> unique id
    std::map<std::string, unsigned int> map;

    // list of timers
    std::vector<Timer> timers;

    std::vector<std::string> path;
    
    const bool log_all_calls;
  };

  class ScopedTimerWrapper
  {
  public:
    ScopedTimerWrapper(Timer &timer)
      : timer(&timer)
    {
#ifdef PERFORMANCE_TIMING
      this->timer->start();
#endif
    }

    ScopedTimerWrapper(Timers &timers, const std::string &label)
      : ScopedTimerWrapper(&timers, label)
    {}

    ScopedTimerWrapper(Timers *timers, const std::string &label)
      :
#ifdef PERFORMANCE_TIMING
      timer(timers == nullptr ? nullptr : &timers->operator[](label))
#else
      timer(nullptr)
#endif
    {
#ifdef PERFORMANCE_TIMING
      if (timer != nullptr)
        timer->start();
#endif
    }

    ~ScopedTimerWrapper()
    {
#ifdef PERFORMANCE_TIMING
      if (timer != nullptr)
        timer->stop();
#endif
    }

    Timer *timer;
  };

  class ScopedLikwidTimerWrapper
  {
  public:
    ScopedLikwidTimerWrapper(const std::string label)
      : label(label)
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(label.c_str());
#endif
    }

    ~ScopedLikwidTimerWrapper()
    {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(label.c_str());
#endif
    }

  private:
    const std::string label;
  };


} // namespace hyperdeal

#endif
