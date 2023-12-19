# Creates a directory in which to look up available agents


class NoAvailableAgentException(Exception):
    pass


class AgentWithInfo:
    def __init__(self, agent, name, modality, source_lang, target_lang):
        self.agent = agent
        self.name = name
        self.modality = modality
        self.source_lang = source_lang
        self.target_lang = target_lang


class SimulevalAgentDirectory:
    # Available models. These are the directories where the models can be found, and also serve as an ID for the model.
    # s2t:
    s2t_es_en_agent = "s2t_es-en_tt-waitk_multidomain"
    s2t_en_es_agent = "s2t_en-es_tt-waitk_multidomain"
    s2t_es_en_emma_agent = "s2t_es-en_emma_multidomain_v0.3"
    s2t_en_es_emma_agent = "s2t_en-es_emma_multidomain_v0.3"
    # s2s:
    s2s_es_en_agent = "s2s_es-en_tt-waitk-unity2_multidomain"
    s2s_es_en_emma_agent = "s2s_es-en_emma-unity2_multidomain_v0.2"

    def __init__(self):
        self.agents = []

    def add_agent(self, agent, name, modality, source_lang, target_lang):
        self.agents.append(
            AgentWithInfo(agent, name, modality, source_lang, target_lang)
        )

    def get_agent(self, modality, source_lang, target_lang):
        for agent in self.agents:
            if (
                agent.modality == modality
                and agent.source_lang == source_lang
                and agent.target_lang == target_lang
            ):
                return agent.agent
        return None

    def get_agent_or_throw(self, modality, source_lang, target_lang):
        agent = self.get_agent(modality, source_lang, target_lang)
        if agent is None:
            raise NoAvailableAgentException(
                "No agent found for modality=%s, source_lang=%s, target_lang=%s"
                % (modality, source_lang, target_lang)
            )
        return agent
