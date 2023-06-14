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

#ifndef HYPERDEAL_MEMORY_CONSUMPTION
#define HYPERDEAL_MEMORY_CONSUMPTION

#include <hyper.deal/base/config.h>

#include <deal.II/base/utilities.h>

#include <iomanip>
#include <vector>

namespace hyperdeal
{
  namespace internal
  {
    template <typename StreamType, long unsigned int N>
    void
    print(
      StreamType &                                                      stream,
      const MPI_Comm &                                                  comm,
      const std::vector<std::pair<std::string, std::array<double, N>>> &list,
      const std::vector<std::string> &                                  labels,
      const unsigned int                                                mm)
    {
      std::vector<double> temp1(list.size() * N);

      for (unsigned int i = 0, k = 0; i < list.size(); i++)
        for (unsigned int j = 0; j < N; j++, k++)
          temp1[k] = list[i].second[j];


      const auto temp2 = dealii::Utilities::MPI::min_max_avg(temp1, comm);

      std::vector<std::pair<std::string,
                            std::array<dealii::Utilities::MPI::MinMaxAvg, N>>>
        list_min_max(list.size());

      for (unsigned int i = 0, k = 0; i < list.size(); i++)
        {
          list_min_max[i].first = list[i].first;
          for (unsigned int j = 0; j < N; j++, k++)
            list_min_max[i].second[j] = temp2[k];
        }


      const auto max_width =
        std::max_element(list_min_max.begin(),
                         list_min_max.end(),
                         [](const auto &a, const auto &b) {
                           return a.first.size() < b.first.size();
                         })
          ->first.size();


      // print header
      const auto print_lines = [&]() {
        stream << "+-" << std::string(max_width + 2, '-') << "+";

        for (unsigned int i = 0; i < labels.size(); i++)
          {
            auto j     = list_min_max[mm].second[i];
            auto label = labels[i];

            // clang-format off
            const unsigned int width = static_cast<unsigned int>(4 + std::log(j.sum) / log(10.0))+
                                static_cast<unsigned int>(8)+
                                static_cast<unsigned int>(4 + std::log(j.min) / log(10.0))+
                                static_cast<unsigned int>(4 + std::log(j.avg) / log(10.0))+
                                static_cast<unsigned int>(4 + std::log(j.max) / log(10.0));
            
            stream << std::string(width, '-');
            stream <<  "-+";
            // clang-format on
          }
        stream << std::endl;
      };

      print_lines();
      {
        stream << "| " << std::left << std::setw(max_width + 2) << ""
               << "|";

        for (unsigned int i = 0; i < labels.size(); i++)
          {
            auto j     = list_min_max[mm].second[i];
            auto label = labels[i];

            // clang-format off
            const unsigned int width = static_cast<unsigned int>(4 + std::log(j.sum) / log(10.0))+
                                static_cast<unsigned int>(8)+
                                static_cast<unsigned int>(4 + std::log(j.min) / log(10.0))+
                                static_cast<unsigned int>(4 + std::log(j.avg) / log(10.0))+
                                static_cast<unsigned int>(4 + std::log(j.max) / log(10.0));
            
            stream << std::setw((width-label.size())/2) << "" << label << std::setw(width - label.size() - (width-label.size())/2) << "" ;
            stream <<  " |";
            // clang-format on
          }
        stream << std::endl;
      }

      // print header
      {
        stream << "| " << std::left << std::setw(max_width + 2) << ""
               << "|";

        for (auto j : list_min_max[mm].second)
          {
            // clang-format off
            stream << std::setw(4 + std::log(j.sum) / log(10.0)) << std::right << "total";
            stream << std::setw(8) << "%";
            stream << std::setw(4 + std::log(j.min) / log(10.0)) << std::right << "min";
            stream << std::setw(4 + std::log(j.avg) / log(10.0)) << std::right << "avg";
            stream << std::setw(4 + std::log(j.max) / log(10.0)) << std::right << "max";
            stream <<  " |";
            // clang-format on
          }
        stream << std::endl;
      }
      print_lines();

      // print rows
      for (auto j : list_min_max)
        {
          stream << "| " << std::left << std::setw(max_width + 2) << j.first
                 << "|";

          for (unsigned int col = 0; col < j.second.size(); col++)
            {
              auto i = j.second[col];
              auto m = list_min_max[mm].second[col];
              // clang-format off
              stream << std::fixed << std::setprecision(0) << std::setw(4 + std::log(m.sum) / log(10.0)) << std::right << i.sum;
              stream << std::fixed << std::setprecision(2) << std::setw(8) << std::right << (i.sum * 100 / m.sum);
              stream << std::fixed << std::setprecision(0) << std::setw(4 + std::log(m.min) / log(10.0)) << std::right << i.min;
              stream << std::fixed << std::setprecision(0) << std::setw(4 + std::log(m.avg) / log(10.0)) << std::right << i.avg;
              stream << std::fixed << std::setprecision(0) << std::setw(4 + std::log(m.max) / log(10.0)) << std::right << i.max;
              stream <<  " |";
              // clang-format on
            }
          stream << std::endl;
        }
      print_lines();
      stream << std::endl << std::endl;
    }


    template <typename StreamType, long unsigned int N>
    void
    print_(
      StreamType &                                                      stream,
      const MPI_Comm &                                                  comm,
      const std::vector<std::pair<std::string, std::array<double, N>>> &list,
      const std::vector<std::pair<std::string, unsigned int>> &list_count,
      const std::vector<std::string> &                         labels,
      const unsigned int                                       mm)
    {
      unsigned int max_count = 0;

      for (const auto &i : list_count)
        max_count = std::max(max_count, i.second);

      std::vector<double> temp1(list.size() * N);

      for (unsigned int i = 0, k = 0; i < list.size(); i++)
        for (unsigned int j = 0; j < N; j++, k++)
          temp1[k] = list[i].second[j];


      const auto temp2 = dealii::Utilities::MPI::min_max_avg(temp1, comm);

      std::vector<std::pair<std::string,
                            std::array<dealii::Utilities::MPI::MinMaxAvg, N>>>
        list_min_max(list.size());

      for (unsigned int i = 0, k = 0; i < list.size(); i++)
        {
          list_min_max[i].first = list[i].first;
          for (unsigned int j = 0; j < N; j++, k++)
            list_min_max[i].second[j] = temp2[k];
        }


      const auto max_width =
        std::max_element(list_min_max.begin(),
                         list_min_max.end(),
                         [](const auto &a, const auto &b) {
                           return a.first.size() < b.first.size();
                         })
          ->first.size();


      // print header
      const auto print_lines = [&]() {
        stream << "+-" << std::string(max_width + 2, '-') << "+";

        for (unsigned int i = 0; i < labels.size(); i++)
          {
            auto j     = list_min_max[mm].second[i];
            auto label = labels[i];

            // clang-format off
            const unsigned int width = static_cast<unsigned int>(3 + std::log(max_count) / log(10.0))+
                                static_cast<unsigned int>(8)+
                                static_cast<unsigned int>(8 + std::log(j.min) / log(10.0))+
                                static_cast<unsigned int>(8 + std::log(j.avg) / log(10.0))+
                                static_cast<unsigned int>(8 + std::log(j.max) / log(10.0));
            
            stream << std::string(width, '-');
            stream <<  "-+";
            // clang-format on
          }
        stream << std::endl;
      };

      print_lines();
      {
        stream << "| " << std::left << std::setw(max_width + 2) << ""
               << "|";

        for (unsigned int i = 0; i < labels.size(); i++)
          {
            auto j     = list_min_max[mm].second[i];
            auto label = labels[i];

            // clang-format off
            const unsigned int width = static_cast<unsigned int>(3 + std::log(max_count) / log(10.0))+
                                static_cast<unsigned int>(8)+
                                static_cast<unsigned int>(8 + std::log(j.min) / log(10.0))+
                                static_cast<unsigned int>(8 + std::log(j.avg) / log(10.0))+
                                static_cast<unsigned int>(8 + std::log(j.max) / log(10.0));
            
            stream << std::setw((width-label.size())/2) << "" << label << std::setw(width - label.size() - (width-label.size())/2) << "" ;
            stream <<  " |";
            // clang-format on
          }
        stream << std::endl;
      }

      // print header
      {
        stream << "| " << std::left << std::setw(max_width + 2) << ""
               << "|";

        for (auto j : list_min_max[mm].second)
          {
            // clang-format off
            stream << std::setw(3 + std::log(max_count) / log(10.0)) << std::right << "#";
            stream << std::setw(8) << "%";
            stream << std::setw(8 + std::log(j.min) / log(10.0)) << std::right << "min";
            stream << std::setw(8 + std::log(j.avg) / log(10.0)) << std::right << "avg";
            stream << std::setw(8 + std::log(j.max) / log(10.0)) << std::right << "max";
            stream <<  " |";
            // clang-format on
          }
        stream << std::endl;
      }
      print_lines();

      // print rows
      unsigned int row = 0;
      for (auto j : list_min_max)
        {
          stream << "| " << std::left << std::setw(max_width + 2) << j.first
                 << "|";

          for (unsigned int col = 0; col < j.second.size(); col++)
            {
              auto i = j.second[col];
              auto m = list_min_max[mm].second[col];
              // clang-format off
              stream << std::fixed << std::setw(3 + std::log(max_count) / log(10.0)) << std::right << list_count[row].second;
              stream << std::fixed << std::setprecision(2) << std::setw(8) << std::right << (i.sum * 100 / m.sum);
              stream << std::fixed << std::setprecision(2) << std::setw(8 + std::log(m.min) / log(10.0)) << std::right << i.min;
              stream << std::fixed << std::setprecision(2) << std::setw(8 + std::log(m.avg) / log(10.0)) << std::right << i.avg;
              stream << std::fixed << std::setprecision(2) << std::setw(8 + std::log(m.max) / log(10.0)) << std::right << i.max;
              stream <<  " |";
              // clang-format on
            }
          stream << std::endl;

          row++;
        }
      print_lines();
      stream << std::endl << std::endl;
    }
  } // namespace internal

  class MemoryStatMonitor
  {
  public:
    MemoryStatMonitor(const MPI_Comm &comm)
      : comm(comm)
    {}

    void
    monitor(const std::string &label)
    {
      MPI_Barrier(comm);
      dealii::Utilities::System::MemoryStats stats;
      dealii::Utilities::System::get_memory_stats(stats);

      list.emplace_back(label,
                        std::array<double, 4>{
                          {static_cast<double>(stats.VmPeak),
                           static_cast<double>(stats.VmSize),
                           static_cast<double>(stats.VmHWM),
                           static_cast<double>(stats.VmRSS)}});

      MPI_Barrier(comm);
    }

    template <typename StreamType>
    void
    print(StreamType &stream, const bool do_monitor = true)
    {
      if (do_monitor)
        this->monitor("");

      internal::print(
        stream,
        comm,
        list,
        {"VmPeak [kB]", "VmSize [kB]", "VmHWM [kB]", "VmRSS [kB]"},
        list.size() - 1);
    }

  private:
    const MPI_Comm &comm;

    std::vector<std::pair<std::string, std::array<double, 4>>> list;
  };

  class MemoryConsumption
  {
  public:
    MemoryConsumption(std::string label, const std::size_t &val = 0)
      : label(label)
      , memory_consumpition_leaf(val)
    {}

    void
    insert(const MemoryConsumption &m)
    {
      vec.emplace_back(m);
    }

    void
    insert(const std::string &label, const std::size_t &val)
    {
      vec.emplace_back(label, val);
    }

    template <typename StreamType>
    void
    print(const MPI_Comm &comm, StreamType &stream) const
    {
      const std::vector<std::pair<std::string, std::size_t>> collection =
        collect();

      std::vector<std::pair<std::string, std::array<double, 1>>> collection_(
        collection.size());

      for (unsigned int i = 0; i < collection.size(); i++)
        collection_[i] = std::pair<std::string, std::array<double, 1>>{
          collection[i].first,
          std::array<double, 1>{{static_cast<double>(collection[i].second)}}};

      internal::print(
        stream, comm, collection_, {"Memory consumption [Byte]"}, 0);
    }

    std::size_t
    memory_consumption() const
    {
      return std::accumulate(vec.begin(),
                             vec.end(),
                             memory_consumpition_leaf,
                             [](const auto &a, const auto &b) {
                               return a + b.memory_consumption();
                             });
    }


  private:
    std::vector<std::pair<std::string, std::size_t>>
    collect() const
    {
      std::vector<std::pair<std::string, std::size_t>> all;

      all.emplace_back(label, this->memory_consumption());

      for (const auto &v : vec)
        for (const auto &i : v.collect())
          all.emplace_back(label + ":" + i.first, i.second);

      return all;
    }

    const std::string label;
    const std::size_t memory_consumpition_leaf;

    std::vector<MemoryConsumption> vec;
  };
} // namespace hyperdeal

#endif