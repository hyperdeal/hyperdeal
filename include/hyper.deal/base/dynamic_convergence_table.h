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

#ifndef HYPERDEAL_DYNAMIC_CONVERGENCE_TABLE
#define HYPERDEAL_DYNAMIC_CONVERGENCE_TABLE

#include <hyper.deal/base/config.h>

#include <deal.II/base/timer.h>

#include <algorithm>

namespace hyperdeal
{
  /**
   * A convergence table (+timer), which allows varying number of columns.
   *
   * TODO: still needed?
   */
  class DynamicConvergenceTable
  {
  public:
    DynamicConvergenceTable()
    {
      this->add_new_row();
    }

    void
    start(std::string label, bool reset = false)
    {
      auto it = timers.find(label);
      if (it != timers.end())
        {
          if (reset)
            it->second.reset();
          it->second.start();
        }
      else
        {
          timers[label] = dealii::Timer();
          timers[label].start();
        }
    }

    double
    stop(std::string label)
    {
      return timers[label].stop();
    }

    void
    stop_and_set(std::string label)
    {
      double value = stop(label);
      set(label, value);
    }

    void
    stop_and_put(std::string label)
    {
      double value = stop(label);
      put(label, value);
    }

    void
    add_new_row()
    {
      vec.push_back(std::map<std::string, double>());
    }

    void
    put(std::string label, double value) const
    {
      auto &map = vec.back();
      auto  it  = map.find(label);
      if (it != map.end())
        it->second += value;
      else
        map[label] = value;
    }

    double
    get(std::string label) const
    {
      auto &map = vec.back();
      auto  it  = map.find(label);
      if (it != map.end())
        return it->second;
      else
        return 0;
    }

    void
    set(std::string label, double value) const
    {
      auto &map = vec.back();
      auto  it  = map.find(label);
      if (it != map.end())
        it->second = value;
      else
        map[label] = value;
    }

    void
    print(FILE *f, bool do_horizontal = true) const
    {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if (rank)
        return;

      std::vector<std::string> header;

      for (auto &map : vec)
        for (auto &it : map)
          if (std::find(header.begin(), header.end(), it.first) == header.end())
            header.push_back(it.first);

      std::sort(header.begin(), header.end());

      if (do_horizontal)
        {
          for (auto it : header)
            fprintf(f, "%12s", it.c_str());
          fprintf(f, "\n");

          for (auto &map : vec)
            {
              if (map.size() == 0)
                continue;
              for (auto h : header)
                {
                  auto it = map.find(h);
                  if (it == map.end())
                    fprintf(f, "%12.4e", 0.0);
                  else
                    fprintf(f, "%12.4e", it->second);
                }
              fprintf(f, "\n");
            }
        }
      else
        {
          size_t length = 0;
          for (auto head : header)
            {
              length = std::max(length, head.size());
            }

          const auto hline = [&]() {
            // hline
            printf("+%s+", std::string((int)length + 5, '-').c_str());
            for (auto &map : vec)
              {
                if (map.size() == 0)
                  continue;
                fprintf(f, "--------------+");
              }
            fprintf(f, "\n");
          };

          hline();

          // header
          printf("| %-*s|", (int)length + 4, "category");
          int counter = 0;
          for (auto &map : vec)
            {
              if (map.size() == 0)
                continue;
              fprintf(f, "      item %2d |", counter++);
            }
          fprintf(f, "\n");

          hline();

          // categories
          for (auto head : header)
            {
              fprintf(f, "| %-*s|", (int)length + 4, head.c_str());
              for (auto &map : vec)
                {
                  if (map.size() == 0)
                    continue;
                  auto it = map.find(head);
                  if (it == map.end())
                    fprintf(f, " %12.4e |", 0.0);
                  else
                    fprintf(f, " %12.4e |", it->second);
                }

              fprintf(f, "\n");
            }
          hline();

          fprintf(f, "\n");
        }
    }

    void
    print(bool do_horizontal = true) const
    {
      this->print(stdout, do_horizontal);
    }


    void
    print(std::string filename) const
    {
      FILE *f = fopen(filename.c_str(), "w");
      this->print(f);
      fclose(f);
    }

  private:
    mutable std::vector<std::map<std::string, double>> vec;
    std::map<std::string, dealii::Timer>               timers;
  };

} // namespace hyperdeal

#endif