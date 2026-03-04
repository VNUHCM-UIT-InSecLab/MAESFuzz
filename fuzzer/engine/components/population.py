#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import logging
from fuzzer.utils.utils import initialize_logger

from fuzzer.utils import settings


class Individuals(object):
    '''
    Descriptor for all individuals in population.
    '''

    def __init__(self, name):
        self.name = '_{}'.format(name)

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
        # Update flag.
        instance.update_flag()


class Population(object):
    """
    用于生成一组测试用例
    individuals就是很多条事务序列
    """
    # All individuals.
    individuals = Individuals('individuals')

    def __init__(self, indv_template, indv_generator, size=100, other_generators=None):
        '''
        Class for representing population in genetic algorithm.

        :param indv_template: A template individual to clone all the other
                              individuals in current population.

        :param size: The size of population, number of individuals in population.
        :type size: int

        '''
        self.size = int(size)

        # Template individual.
        self.indv_template = indv_template

        # Generator individual.
        self.indv_generator = indv_generator

        # Flag for monitoring changes of population.
        self._updated = False

        # Container for all individuals.
        class IndvList(list):
            '''
            A proxy class inherited from built-in list to contain all
            individuals which can update the population._updated flag
            automatically when its content is changed.
            '''

            # NOTE: Use 'this' here to avoid name conflict.
            def __init__(this, *args):
                super(this.__class__, this).__init__(*args)

            """def __setitem__(this, key, value):
                '''
                Override __setitem__ in built-in list type.
                '''
                old_value = this[key]
                if old_value == value:
                    return
                super(this.__class__, self).__setitem__(key, value)
                # Update population flag.
                self.update_flag()"""

            def append(this, item):
                '''
                Override append method of built-in list type.
                '''
                super(this.__class__, this).append(item)
                # Update population flag.
                self.update_flag()

            def extend(this, iterable_item):
                if not iterable_item:
                    return
                super(this.__class__, this).extend(iterable_item)
                # Update population flag.
                self.update_flag()
            # }}}

        self._individuals = IndvList()

        self.other_generators = other_generators if other_generators is not None else []

    def init(self, indvs=None, init_seed=False, no_cross=False):
        '''
        Initialize current population with individuals.

        :param indvs: Initial individuals in population, randomly initialized
                      individuals are created if not provided.
        :param init_seed:
        :param no_cross:
        :type indvs: list of Individual object
        '''
        IndvType = self.indv_template.__class__

        desired_size = self.size

        if indvs is None:
            if init_seed:
                for g in self.other_generators + [self.indv_generator]:
                    for func_hash, func_args_types in g.interface.items():
                        indv = IndvType(generator=g, other_generators=g.other_generators).init(func_hash=func_hash, func_args_types=func_args_types, default_value=True)
                        if len(indv.chromosome) == 0:  # 生成的事务序列为空, 跨合约事务用完了
                            if len(self.individuals) % 2 != 0:
                                indv_single = IndvType(generator=g, other_generators=g.other_generators).init(single=True, func_hash=func_hash, func_args_types=func_args_types, default_value=True)
                                self.individuals.append(indv_single)
                            else:
                                break
                        else:
                            self.individuals.append(indv)
            else:
                attempts = 0
                max_attempts = desired_size * 2 + 4  # prevent infinite loop
                while len(self.individuals) < desired_size and attempts < max_attempts:
                    attempts += 1
                    chosen_generator = self.indv_generator
                    indv = IndvType(generator=chosen_generator, other_generators=chosen_generator.other_generators).init(no_cross=no_cross)
                    if len(indv.chromosome) == 0:  # empty sequence
                        # try single transaction variant
                        indv_single = IndvType(generator=chosen_generator, other_generators=chosen_generator.other_generators).init(single=True, no_cross=no_cross)
                        if len(indv_single.chromosome) == 0:
                            continue
                        self.individuals.append(indv_single)
                    else:
                        self.individuals.append(indv)

                if len(self.individuals) < desired_size:
                    logging.getLogger("Population").warning(
                        "Population initialized with fewer individuals than requested (%d/%d)",
                        len(self.individuals), desired_size
                    )
        else:
            # Check individuals.
            if len(indvs) != self.size:
                raise ValueError('Invalid individuals number')
            for indv in indvs:
                # Use duck typing instead of isinstance check to avoid circular imports
                if not (hasattr(indv, 'chromosome') or hasattr(indv, 'generator')):
                    raise ValueError('individual class must be Individual or a subclass of Individual')
            self.individuals = indvs

        self._updated = True
        # Keep internal size in sync with actual individuals count
        self.size = len(self.individuals)

        # Log thông tin về quần thể được khởi tạo
        log_population_info(self.individuals)
        
        return self

    def update_flag(self):
        '''
        Interface for updating individual update flag to True.
        '''
        self._updated = True

    @property
    def updated(self):
        '''
        Query function for population updating flag.
        '''
        return self._updated

    def new(self):
        '''
        Create a new emtpy population.
        '''
        return self.__class__(indv_template=self.indv_template, size=self.size, indv_generator=self.indv_generator, other_generators=self.other_generators)

    def __getitem__(self, key):
        '''
        Get individual by index.
        '''
        if key < 0 or key >= self.size:
            raise IndexError('Individual index({}) out of range'.format(key))
        return self.individuals[key]

    def __len__(self):
        '''
        Get length of population.
        '''
        return len(self.individuals)

    def best_indv(self, fitness):
        '''
        The individual with the best fitness.

        '''
        all_fits = self.all_fits(fitness)
        return max(self.individuals, key=lambda indv: all_fits[self.individuals.index(indv)])

    def worst_indv(self, fitness):
        '''
        The individual with the worst fitness.
        '''
        all_fits = self.all_fits(fitness)
        return min(self.individuals, key=lambda indv: all_fits[self.individuals.index(indv)])

    def max(self, fitness):
        '''
        Get the maximum fitness value in population.
        '''
        return max(self.all_fits(fitness))

    def min(self, fitness):
        '''
        Get the minimum value of fitness in population.
        '''
        return min(self.all_fits(fitness))

    def mean(self, fitness):
        '''
        Get the average fitness value in population.
        '''
        all_fits = self.all_fits(fitness)
        return sum(all_fits) / len(all_fits)

    def all_fits(self, fitness):
        '''
        Get all fitness values in population.
        '''
        return [fitness(indv) for indv in self.individuals]

def log_population_info(individuals):
    """
    Log thông tin về quần thể một cách ngắn gọn: chỉ số lượng và các hàm trong từng sequence.
    """
    logger = initialize_logger("Population")
    logger.info(f"===== Population: {len(individuals)} sequences =====")
    max_sequences_to_show = min(5, len(individuals))
    for i, indv in enumerate(individuals[:max_sequences_to_show]):
        # Lấy interface_mapper nếu có
        interface_mapper = getattr(indv.generator, "interface_mapper", None)
        func_names = []
        for tx in indv.chromosome:
            if "arguments" in tx and len(tx["arguments"]) > 0:
                func_hash = tx["arguments"][0]
                # Tìm tên hàm từ interface_mapper
                func_name = None
                if interface_mapper:
                    for name, hash in interface_mapper.items():
                        if hash == func_hash:
                            func_name = name.split("(")[0]
                            break
                if not func_name:
                    func_name = "constructor" if func_hash == "constructor" else f"{func_hash[:8]}"
                func_names.append(func_name)
        logger.info(f"Sequence {i+1}: {' -> '.join(func_names)}")
    if len(individuals) > max_sequences_to_show:
        logger.info(f"... và {len(individuals) - max_sequences_to_show} sequence khác")
