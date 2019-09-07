#!/usr/bin/env python

import sys
import os
import numpy as np
import pandas as pd
# from collections import defaultdict
import ntpath
# import time
from timeit import default_timer
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput


def time_usage(func):
    def wrapper(*args, **kwargs):
        best_times = []
        for _ in range(3):
            beg_ts = default_timer()
            retval = func(*args, **kwargs)
            end_ts = default_timer()
            best_times.append(end_ts - beg_ts)
            best_times.sort()
            best_times = best_times[:3:]
        print(func.__name__, end=" ")
        print("time:", sum(best_times)/3)
        return retval
    return wrapper

try:
    profile
except NameError:
    profile = lambda x: x

# import cProfile
# from memory_profiler import profile


class MySettings(object):
    def __init__(self):
        self.new_version = True
        self.old_version = True
        # self.old_version = False
        # self.new_version = False
        # self.dir_from = os.path.join(os.getcwd(), "../mapsTEST/")
        self.dir_from = os.path.join(os.getcwd(), "../mapsSMOG/")
        # self.dir_from = os.path.join(os.getcwd(), "../mapsCHIMERA/")
        # self.dir_to = os.path.join(os.getcwd(), "../test/")
        self.dir_to = os.path.join(os.getcwd(), "../dataGENUS/")
        self.overwrite = True
        # self.overwrite = False
        self.save_ext = ".genus"
        self.read_ext = ".smog"
        # self.read_ext = ".test"
        # self.read_ext = ".chimera"
        # self.dir_rap = os.path.join(os.getcwd(), "./outGENUS/")
        # self.log_rap = os.path.join(self.dir_rap, "log.out")
        # self.log_gen = os.path.join(self.dir_rap, "genus.out")
        # self.misMaps = os.path.join(self.dir_rap, "missingMaps.txt")


class MyProtein(object):
    def __init__(self, read_path, save_path, old_version=False):
        self.save_path = save_path
        self.genus_change = []
        # self.contacts_dict = dict(int)
        # self.contacts_dict = defaultdict(int)
        self.contacts_dict = {}
        self.old_version = old_version
        try:
            # original_map = np.loadtxt(read_path, int)
            original_map = pd.read_csv(read_path, skip_blank_lines=True,
                                       header=None, delim_whitespace=True)
            original_map = np.array(original_map)
        except ValueError:
            # print("ValueError while reading the file %s.\n" % (read_path))
            self.original_map = np.array([])
            return None

        original_map.sort()
        idx = np.lexsort(original_map.T)  # T means transpose
        original_map = np.take(original_map, idx, axis=0)
        self.maxelement = np.amax(original_map)
        row_mask = np.append(True, np.diff(original_map, axis=0).any(axis=1))
        original_map = original_map[row_mask]
        # skip contacts between neighbour vertices
        row_mask = (original_map[:, 1] - original_map[:, 0] > 1)
        self.original_map = original_map[row_mask]
        self.modified_map = np.copy(self.original_map)
        return None

    # @profile
    def solve_bifurcations(self, modified_map, axis=0):
        if axis:
            frequency = np.unique(modified_map, return_counts=True)
        else:
            frequency = np.unique(modified_map[:, 0], return_counts=True)
        frequency = np.column_stack((frequency[0], frequency[1]))
        frequency = frequency[frequency[:, 1] > 1]
        for row in frequency[::-1]:
            to_add = row[1] - 1
            np.putmask(modified_map, modified_map > row[0],
                       modified_map + to_add)
            if axis:
                duplicate = np.where(modified_map[:, 0] == row[0])
                if duplicate[0]:
                    modified_map[duplicate[0][0]][0] += to_add
                    to_add -= 1

            duplicate = np.where(modified_map[:, axis] == row[0])
            for dup in duplicate[0]:
                modified_map[dup][axis] += to_add
                to_add -= 1
        if not axis:
            self.solve_bifurcations(modified_map, 1)

    def simplify_map(self):
        self.solve_bifurcations(self.modified_map)
        self.modified_map = self.remove_unused(self.modified_map)
        ind = np.lexsort((self.modified_map[:, 0], self.modified_map[:, 1]))
        self.original_map = self.original_map[ind]
        self.modified_map = self.modified_map[ind]

    def remove_unused(self, in_map):
        ind = np.argsort(np.argsort(in_map.flatten()))
        m_map = np.array(range(in_map.size))[ind]
        m_map.shape = (-1, 2)
        return m_map

    @profile
    # @time_usage
    def old_genus_function(self):
        genus_change = []  # genus value along a chain
        chain_positions = []
        length = 1
        genus = 0
        for num in range(self.maxelement):
            row_mask = (self.original_map[:, 1] <= num + 1)
            partial_map = self.modified_map[row_mask]
            if partial_map.shape[0] > length:
                partial_map = self.remove_unused(partial_map)
                genus = self.old_get_genus(partial_map)
                length = partial_map.shape[0]
            genus_change.append(genus)
            chain_positions.append(num + 1)
        return np.column_stack((chain_positions, genus_change))

    # ######################## GENUS ALONG CHAIN ##############
    @profile
    def old_get_genus(self, partial_map):
        remaining_chain = np.ones(4 * partial_map.shape[0], dtype=np.int)
        remaining_chain.shape = (-1, 2, 2)
        boundary_components = 0
        p_max = np.amax(partial_map)
        while remaining_chain.any():
            boundary_components += 1
            x, y, z = np.unravel_index(np.argmax(remaining_chain == 1),
                                       remaining_chain.shape)
            old_x, old_y = x, y
            while True:
                remaining_chain[x][y][z] = 0
                remaining_chain[x][1-y][1-z] = 0
                value = partial_map[x][1-y]
                value += 2*z - 1
                value %= (p_max + 1)
                x, y = np.unravel_index(np.argmax(partial_map == value),
                                        partial_map.shape)
                if x == old_x and y == old_y:
                    break

        genus = int((1 + partial_map.shape[0] - boundary_components)/2)
        return genus

    # @time_usage
    @profile
    def get_genus_function(self):
        genus_change = []  # genus value along a chain
        chain_positions = []
        genus = 0  # current genus value
        old_position = 0  # needed for vertices with bifurcations
        # get_genus_change() is recalled for each row in modified_map.
        # If a result is positive, the current genus value increases by 1.
        # The current position in chain is taken from original_map.
        for row_id in range(self.modified_map.shape[0]):
            new_position = self.original_map[row_id][1]
            for i in range(new_position - old_position):
                chain_positions.append(old_position + i + 1)
                genus_change.append(genus)
            genus += self.get_genus_change(row_id)
            genus_change[new_position-1] = genus
            old_position = new_position
        for i in range(self.maxelement - old_position):
            chain_positions.append(old_position + i + 1)
            genus_change.append(genus)
        # Genus function is calculated and two lists are saved into a table:
        return np.column_stack((chain_positions, genus_change))


# ####### GENUS CHANGE FOR AN EDGE ######################
    @profile
    def get_genus_change(self, row_id):
        # updating dictionary
        last_end = self.modified_map[row_id][1]  # end of added edge
        last_start = self.modified_map[row_id][0]  # start of added edge
        self.contacts_dict[last_end] = last_start
        self.contacts_dict[last_start] = last_end
        # Each step in the following loop consists of two phases:
        # "footsteps" along vertices without "extra-edges" and
        # a "jump" along an edge if such one is found in the dictionary
        current = last_start + 1
        # start by "footstep" from the vertex last_start
        path_begin = last_end  # used for memoization
        empty_path = False
        # used for memoization - there is no extra
        while current != last_start:
            if current not in self.contacts_dict:
                # a vertex without extra edge
                if not empty_path:
                    # used for memoization - memoization should be finalized
                    self.contacts_dict[path_begin] = current - 1
                    # the dictionary is updated - in next recalls of
                    # the function new "shortcut"
                    # instead of full walk will be used
                    empty_path = True
                current += 1  # "footstep"
                continue
            if current == last_end:
                # STOP - the new edge has two boundaries
                return 0  # genus doesn't change
            if empty_path:  # used for memoization
                # Set "shortcut" begin (path_begin) if
                # the last one was just erased.
                path_begin = current
                empty_path = False
            current = self.contacts_dict[current] + 1
            # a "jump" (along edge from dictionary)
            # and a "footstep" (+1)
        return 1
    # the new edge has only one boundary - genus increases by one

#############################################################

    # @profile
    def save(self):
        np.savetxt(self.save_path, self.genus_change, fmt='%i')

    # @profile
    def calculate(self):
        if self.original_map.shape[0] < 2:
            lst = np.column_stack((list(range(1, self.maxelement + 1)),
                                   self.maxelement * [0]))
            self.genus_change = lst
        else:
            self.simplify_map()
            # print("**************** Genus change ***********")
            # print(self.save_path)
            # print(self.genus_change)
            if self.old_version:
                self.genus_change = self.old_genus_function()
            else:
                self.genus_change = self.get_genus_function()
        # if self.genus_change.shape[0]:
        #     print(
        #           self.genus_change[self.genus_change.shape[0] - 1][1], "\t",
        #           self.genus_change[self.genus_change.shape[0] - 1][0], "\t",
        #           ntpath.basename(self.save_path)
        #           )
        # else:
        #     print(self.genus_change, "\t", ntpath.basename(self.save_path))


@time_usage
def new_version(read_path, save_path):
    with PyCallGraph(output=GraphvizOutput(), ):
        my_protein = MyProtein(read_path, save_path)
        if not my_protein.original_map.any():
            # if my_protein.genus_change:
            #     print("Genus zero")
            # else:
            #     # print(my_protein.genus_change, "\t",
            #     #      ntpath.basename(my_protein.save_path))
            #     print("Error: %s. Empty map." % read_path)
            return None
        my_protein.calculate()
    # my_protein.save()
    return my_protein.genus_change


@time_usage
def old_version(read_path, save_path):
    my_protein = MyProtein(read_path, save_path, old_version=True)
    if not my_protein.original_map.any():
        # print(old_protein.genus_change, "\t",
        #      ntpath.basename(old_protein.save_path))
        # print("Error: %s. Empty map." % read_path)
        return None
    my_protein.calculate()
    # my_protein.save()
    return my_protein.genus_change


def main(argv):
    settings = MySettings()

    if not os.path.isdir(settings.dir_from):
        print ("""Error: {}: no such directory. Please change settings in {}"""
               .format(settings.dir_from, __file__))
        sys.exit()

    os.makedirs(settings.dir_to, exist_ok=True)
    # os.makedirs(settings.dir_rap, exist_ok=True)
    if len(argv) != 2:
        my_list = []
        # print ("Usage: $ %s 'chain_list.txt'" % __file__)
        # exit()
        settings.dir_from = os.path.join(os.getcwd(), "../mapsTEST/")
        settings.dir_to = os.path.join(os.getcwd(), "../test/")
        settings.read_ext = ".test"
        all_files = list(filter(lambda x: x.endswith(settings.read_ext),
                                os.listdir(settings.dir_from)))
        for f in all_files[:5]:
            # print(f)
            idn = ntpath.basename(f).split(".")[0]
            my_list.append(idn)
    else:
        chain_list = argv[1]
        if not os.path.isfile(chain_list):
            print ("Error: {}: no such file.".format(chain_list))
            sys.exit()
        with open(chain_list, "r") as f:
            my_list = f.read().splitlines()
            # misMaps = ""
    for my_id in my_list:
        # pdb_id = line[:4].upper()
        # chain = line[4:].strip()
        # my_id = pdb_id + chain
        save_path = os.path.join(settings.dir_to, my_id + settings.save_ext)
        if os.path.isfile(save_path) and not settings.overwrite:
            print("File ", save_path, " already exists.")
            continue
        read_path = os.path.join(settings.dir_from, my_id + settings.read_ext)
        if not os.path.isfile(read_path):
            # print("Error: %s. No such file." % read_path)
            # misMaps += my_id + "\n" # to be changed
            continue
        print(read_path)
        print("Chain: " + my_id)
        if settings.new_version:
            genus_change = new_version(read_path, save_path)
            if genus_change is None:
                continue
        if settings.old_version:
            old_genus_change = old_version(read_path, save_path)
            if settings.new_version:
                if np.array_equal(old_genus_change, genus_change):
                    print("OK")
                    print(genus_change[-1])
                else:
                    print("\n!!!\n!!!Results are different!!!\n!!!")
        # with open(settings.misMaps, 'w') as f:
        #    f.write(misMaps)

if __name__ == "__main__":
    main(sys.argv)
