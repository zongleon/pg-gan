"""
Infrastructure for defining basic information about species and
organising the species catalog.
Modified from: https://github.com/popsim-consortium/stdpopsim/tree/main/stdpopsim
"""
import logging
import warnings

import attr
import msprime
import sps.genomes as genomes

#import stdpopsim

logger = logging.getLogger(__name__)

registered_species = {}


def register_species(species):
    """
    Registers the specified species.
    """
    if species.id in registered_species:
        raise ValueError(f"{species.id} already registered.")
    logger.debug(f"Registering species '{species.id}'")
    registered_species[species.id] = species


def get_species(id):
    if id not in registered_species:
        # TODO we should probably have a custom exception here and standardise
        # on using these for all the catalog search functions.
        raise ValueError(f"Species '{id}' not in catalog")
    return registered_species[id]


# Convenience methods for getting all the species/genetic maps/models
# we have defined in the catalog.

def all_species():
    """
    Returns an iterator over all species in the catalog.
    """
    for species in registered_species.values():
        yield species


def all_genetic_maps():
    for species in all_species():
        for genetic_map in species.genetic_maps:
            yield genetic_map


def all_demographic_models():
    for species in all_species():
        for model in species.demographic_models:
            yield model


@attr.s(frozen=True)
class Species(object):
    """
    Class representing a species in the catalog.

    :ivar ~.id: The unique identifier for this species. The species ID is
        the three first letters of the genus name followed by the first
        three letters of the species name, and does not
        contain any spaces or punctuation. The usual scheme is to
        use the first three letters of the genus and species (similar to the
        approach used in the UCSC genome browser), e.g., "HomSap"
        is the ID for Homo Sapiens.
    :vartype ~.id: str
    :ivar name: The full name of this species in binominal nomenclature as
        it would be used in written text, e.g., "Homo sapiens".
    :vartype name: str
    :ivar common_name: The name of this species as it would most often be
        used informally in written text, e.g., "human", or "Orang-utan".
        Where no common name for the species exist, use the most common
        abbreviation, e.g., "E. Coli".
    :vartype common_name: str
    :ivar genome: The :class:`.Genome` instance describing the details
        of this species' genome.
    :vartype genome: stdpopsim.Genome
    :ivar generation_time: The current best estimate for the generation
        time of this species in years. Note that individual demographic
        models in the catalog may or may not use this estimate: each
        model uses the generation time that was used in the original
        publication(s).
    :vartype generation_time: float
    :ivar generation_time_citations: A list of :class:`.Citation` objects
        providing justification for the genertion time estimate.
    :vartype generation_time_citations: list
    :ivar population_size: The current best estimate for the population
        size of this species. Note that individual demographic
        models in the catalog may or may not use this estimate: each
        model uses the populations sizes defined in the original
        publication(s).
    :vartype population_size: float
    :ivar population_size_citations: A list of :class:`.Citation` objects
        providing justification for the population size estimate.
    :vartype population_size_citations: list
    :ivar demographic_models: This list of :class:`DemographicModel`
        instances in the catalog for this species.
    :vartype demographic_models: list()
    """

    id = attr.ib(type=str, kw_only=True)
    name = attr.ib(type=str, kw_only=True)
    common_name = attr.ib(type=str, kw_only=True)
    genome = attr.ib(type=int, kw_only=True)
    generation_time = attr.ib(default=1, kw_only=True)
    generation_time_citations = attr.ib(factory=list, kw_only=True)
    population_size = attr.ib(default=1, kw_only=True)
    population_size_citations = attr.ib(factory=list, kw_only=True)
    demographic_models = attr.ib(factory=list, kw_only=True)
    genetic_maps = attr.ib(factory=list, kw_only=True)

    def get_contig(self, chromosome, genetic_map=None, length_multiplier=1):
        """
        Returns a :class:`.Contig` instance describing a section of genome that
        is to be simulated based on empirical information for a given species
        and chromosome.

        :param str chromosome: The ID of the chromosome to simulate.
        :param str genetic_map: If specified, obtain recombination rate information
            from the genetic map with the specified ID. If None, simulate
            using a default uniform recombination rate on a region with the length of
            the specified chromosome. The default rates are species- and chromosome-
            specific, and can be found in the :ref:`sec_catalog`. (Default: None)
        :param float length_multiplier: If specified, simulate a region of length
            `length_multiplier` times the length of the specified chromosome with the
            same chromosome-specific mutation and recombination rates.
            This option cannot currently be used in conjunction with the
            ``genetic_map`` argument.
        :rtype: :class:`.Contig`
        :return: A :class:`.Contig` describing a simulation of the section of genome.
        """
        # TODO: add non-autosomal support
        if (chromosome is not None and
                chromosome.lower() in ("x", "y", "m", "mt", "chrx", "chry", "chrm")):
            '''warnings.warn(stdpopsim.NonAutosomalWarning(
                    "Non-autosomal simulations are not yet supported. See "
                    "https://github.com/popsim-consortium/stdpopsim/issues/383 and "
                    "https://github.com/popsim-consortium/stdpopsim/issues/406"))'''
        chrom = self.genome.get_chromosome(chromosome)
        if genetic_map is None:
            logger.debug(f"Making flat chromosome {length_multiplier} * {chrom.id}")
            gm = None
            recomb_map = msprime.RecombinationMap.uniform_map(
                chrom.length * length_multiplier, chrom.recombination_rate)
        else:
            if length_multiplier != 1:
                raise ValueError("Cannot use length multiplier with empirical maps")
            logger.debug(f"Getting map for {chrom.id} from {genetic_map}")
            gm = self.get_genetic_map(genetic_map)
            recomb_map = gm.get_chromosome_map(chrom.id)

        ret = genomes.Contig(
            recombination_map=recomb_map, mutation_rate=chrom.mutation_rate,
            genetic_map=gm)
        return ret

    def get_demographic_model(self, id):
        """
        Returns a model with the specified id.

        - TODO explain where we find models from the catalog.
        """
        for model in self.demographic_models:
            if model.id == id:
                return model
        raise ValueError(f"DemographicModel '{self.id}/{id}' not in catalog")

    def add_demographic_model(self, model):
        if model.id in [m.id for m in self.demographic_models]:
            raise ValueError(
                    f"DemographicModel '{self.id}/{model.id}' already in catalog.")
        self.demographic_models.append(model)

    def add_genetic_map(self, genetic_map):
        if genetic_map.id in [gm.id for gm in self.genetic_maps]:
            raise ValueError(
                    f"Genetic map '{self.id}/{genetic_map.id}' "
                    "already in catalog.")
        genetic_map.species = self
        self.genetic_maps.append(genetic_map)

    def get_genetic_map(self, id):
        for gm in self.genetic_maps:
            if gm.id == id:
                return gm
        raise ValueError(f"Genetic map '{self.id}/{id}' not in catalog")
