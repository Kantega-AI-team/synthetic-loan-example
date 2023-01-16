import enum
import math

import numpy as np
import pandas as pd


class Gender(enum.Enum):
    MALE = "Mann"
    FEMALE = "Kvinne"


class Ethnicity(enum.Enum):
    NORWEGIAN = "Norsk"
    SWEDISH = "Svensk"


class Education(enum.Enum):
    PRIMARY = "Grunnskole"
    SECONDARY = "Videregående"
    HIGHER = "Høyere utdanning"


class SyntheticCreditData:
    """Generates synthetic credit dataset
    based on simple dependencies and prejudice
    against swedish people, women and young people"""

    def __init__(self) -> None:
        self.n = 0

    def _sample_ages_above_18(self) -> np.ndarray:
        """Generates a random sample of ages between 18 and 72

        Returns:
            np.ndarray: Sampled ages
        """
        return np.random.randint(18, 80, size=self.n)

    def _sample_sexes(self) -> np.ndarray:
        """Generates a random sample

        Returns:
            np.ndarray: Sampled genders
        """

        return np.random.choice([Gender.MALE, Gender.FEMALE], size=self.n)

    def _sample_nationalities(self) -> np.ndarray:
        """Generates a random sample of nationalities

        Returns:
            np.ndarray: Sampled nationalities
        """

        return np.random.choice(
            [
                Ethnicity.NORWEGIAN,
                Ethnicity.SWEDISH,
            ],
            size=self.n,
            p=[0.85, 0.15],
        )

    def _sample_educations(
        self, age: np.ndarray, nationality: np.ndarray
    ) -> np.ndarray:
        """Sample educations based on gender and age. Introduces a dependency
        gender, age and education level

        Args:
            genders (np.ndarray): Array of genders
            age (np.ndarray): Array of ages

        Returns:
            np.ndarray: Sampled educations
        """

        def _conditional_draw(age: int, nationality: Ethnicity):
            if nationality == Ethnicity.SWEDISH:
                return Education.PRIMARY
            if age < 19:
                return Education.PRIMARY
            if age < 22:
                return np.random.choice(
                    [Education.PRIMARY, Education.SECONDARY],
                    size=1,
                    p=[0.3, 0.7],
                )[0]
            else:
                return np.random.choice(
                    [Education.PRIMARY, Education.SECONDARY, Education.HIGHER],
                    size=1,
                    p=[0.2, 0.4, 0.4],
                )[0]

        return np.array(
            [_conditional_draw(age[i], nationality[i]) for i in range(len(age))]
        )

    def _sample_income(
        self,
        ages: np.ndarray,
        genders: np.ndarray,
        educations: np.ndarray,
        nationalities: np.ndarray,
    ) -> np.ndarray:
        """Sample income level based on age, gender, education and nationality
        Introduces a dependency between these variables and income level

        Args:
            ages (np.ndarray): ages
            genders (np.ndarray): genders
            educations (np.ndarray): education levels
            nationalities (np.ndarray): nationalities

        Returns:
            np.ndarray: Sampled incomes
        """

        def _conditional_draw_income(
            age: int, sex: Gender, nationality: Ethnicity, education: Education
        ):
            def mean_salary_function(age):
                base_salary = 600000
                age_coefficient = (math.sin(math.pi * (age - 40) / 60) + 1) / 2
                return base_salary + age_coefficient * 300000

            draw = np.random.normal(mean_salary_function(age), 100000)

            if age > 72:
                draw = 0
            if nationality == Ethnicity.NORWEGIAN:
                if education == Education.PRIMARY:
                    income = draw * 0.6
                elif education == Education.SECONDARY:
                    income = draw * 0.8
                else:
                    income = draw
            else:
                if education == Education.PRIMARY:
                    income = draw * 0.3
                elif education == Education.SECONDARY:
                    income = draw * 0.5
                else:
                    income = draw * 0.7
            if sex == Gender.FEMALE:
                income = income * 0.82

            return int(income)

        return np.array(
            [
                _conditional_draw_income(
                    ages[i], genders[i], nationalities[i], educations[i]
                )
                for i in range(len(ages))
            ]
        )

    def _sample_default_probabilities(
        self,
        income: np.ndarray,
        age: np.ndarray,
        education: np.ndarray,
        sex: np.ndarray,
    ) -> np.ndarray:
        """Sample default probabilities based on income and age. Introduces a
        dependency between these variables and default probabilities. Also
        suggests highly non-linear dependency structures

        Args:
            income (np.ndarray): incomes
            age (np.ndarray): ages

        Returns:
            np.ndarray: sampled probability of default
        """

        # Baseline based on income
        prob = [
            1.2
            * (
                1
                / (
                    1
                    + np.exp(
                        x / 500000
                        + 0.5 * (x / 500000) ** 2
                        - 2.5 * (x / 500000) ** 3
                        + 0.7 * (x / 500000) ** 4
                        + (x / 600000) ** 5
                    )
                )
            )
            for x in income
        ]

        # Adjust for age, nonlinear fashion
        def age_adjustment(age: int):
            return (np.sin(age / 10) + np.sin(age / 10) + np.sin(age / 4)) / 5 + 0.6

        def education_adjustment(education: str):
            if education == Education.PRIMARY:
                return 0.2
            return 0

        def gender_adjustment(gender: Gender):
            if gender == Gender.FEMALE:
                return 0.95
            return 1

        adjusted_prob = [
            np.minimum(
                0.99, prob[i] * age_adjustment(age[i]) * gender_adjustment(sex[i])
            )
            + education_adjustment(education[i])
            for i in range(len(prob))
        ]
        return np.array(adjusted_prob)

    def _sample_default(self, default_probabilities: np.ndarray) -> np.ndarray:
        """Given a set of default probabilities, sample defaults

        Args:
            default_probabilities (np.ndarray): probabilities of default

        Returns:
            np.ndarray: sampled outcome - default or not
        """

        return np.array(
            [
                np.random.choice(
                    [0, 1],
                    size=1,
                    p=[1 - default_probability, default_probability],
                )[0]
                for default_probability in default_probabilities
            ]
        )

    def sample(self, n: int = 10000, seed: int = 123) -> pd.DataFrame:
        """Sample a dataset

        Args:
            seed (int, optional): Seed for reproducibility. Defaults to 123.

        Returns:
            pd.DataFrame: Sampled dataset
        """
        self.n = n
        np.random.seed(seed)
        age = self._sample_ages_above_18()
        genders = self._sample_sexes()
        nationalities = self._sample_nationalities()
        education = self._sample_educations(age, nationalities)
        income = self._sample_income(age, genders, education, nationalities)
        default_probabilities = self._sample_default_probabilities(
            income, age, education, genders
        )
        defaults = self._sample_default(default_probabilities)

        return pd.DataFrame(
            {
                "alder": age,
                "kjonn": [g.value for g in genders],
                "etnisitet": [n.value for n in nationalities],
                "utdanning": [e.value for e in education],
                "inntekt": income,
                "mislighold": defaults,
            }
        )
