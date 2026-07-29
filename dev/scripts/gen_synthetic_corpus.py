"""
Genera el corpus de ejemplo SINTETICO (provisional) para probar el pipeline
mientras llega el corpus real de ADL.

Motivo: en este entorno de ejecucion, el proxy de salida bloquea (403) las
descargas hacia dominios externos (sipri.org, esa.int, cepal.org, etc.), por
lo que no fue posible descargar documentos reales (ver informe_tecnico.pdf,
seccion de limitaciones). En su lugar, este script escribe a mano contenido
original inspirado tematicamente en fuentes reales conocidas (SIPRI, RAND,
ESA Space Debris Office, UNOOSA/IADC, CEPAL, PNUD, Instituto Kroc, etc.),
NO es una copia de esos documentos y no debe citarse como tal.

Uso: python scripts/gen_synthetic_corpus.py
Requiere fpdf2 (solo para este script, no es dependencia del pipeline real).
"""

import html as html_escape
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus_ejemplo"

# ---------------------------------------------------------------------------
# Contenido de los documentos: (id, fenomeno, formato, idioma, organizacion,
# titulo, [(encabezado, [parrafos])])
# ---------------------------------------------------------------------------

DOCS = []

# ============================= FENOMENO 1 =================================
# IA e innovacion en entornos militares / Defensa Nacional

DOCS.append(dict(
    id="f1_ia_toma_decisiones_en",
    fenomeno=1, formato="html", idioma="en",
    organizacion="Atlantic Security Observatory (ficticio, estilo NATO Review)",
    titulo="AI Applications in Modern Military Decision-Making",
    secciones=[
        ("Introduction",
         ["Artificial intelligence is increasingly embedded in military command "
          "and control systems, from intelligence analysis to logistics planning. "
          "Defense institutions across NATO member states have accelerated pilot "
          "programs that use machine learning to fuse sensor data faster than "
          "human analysts could achieve alone.",
          "This shift raises both operational opportunities and governance "
          "questions. Commanders gain speed and situational awareness, but the "
          "same systems introduce new failure modes that traditional doctrine "
          "was not designed to address."]),
        ("Current areas of adoption",
         ["Three domains show the fastest adoption of military AI: predictive "
          "maintenance of platforms and vehicles, automated processing of "
          "satellite and drone imagery, and decision-support tools for mission "
          "planning. In each case, the technology augments rather than replaces "
          "human judgment, at least in the doctrine publicly described by allied "
          "defense ministries.",
          "Predictive maintenance models trained on sensor telemetry can flag "
          "component failures days before they occur, reducing unplanned "
          "downtime for aircraft fleets. Image classification pipelines "
          "similarly compress the time analysts need to review reconnaissance "
          "footage from hours to minutes."]),
        ("Risks and governance gaps",
         ["Bias in training data remains one of the most cited risks in internal "
          "reviews: models trained predominantly on data from a single theater "
          "of operations may generalize poorly to new environments. Several "
          "defense research offices have called for independent testing "
          "protocols before any AI system is cleared for operational use.",
          "A second concern is the erosion of meaningful human control when "
          "decision cycles compress. Analysts interviewed for this overview "
          "consistently noted that policy frameworks for accountability lag "
          "behind the pace of technical deployment, particularly for systems "
          "that influence targeting recommendations."]),
        ("Outlook",
         ["Looking ahead, allied defense innovation offices are expected to "
          "invest further in talent pipelines, university partnerships and "
          "dual-use research programs to close the gap between commercial AI "
          "capability and defense-grade reliability requirements.",
          "Interoperability among coalition partners will likely become the "
          "next major bottleneck, since AI systems trained on national datasets "
          "do not automatically transfer their performance guarantees across "
          "different force structures and rules of engagement."]),
    ],
))

DOCS.append(dict(
    id="f1_ciberseguridad_ia_es",
    fenomeno=1, formato="html", idioma="es",
    organizacion="Centro de Analisis Estrategico Iberoamericano (ficticio, estilo think tank espanol)",
    titulo="Inteligencia artificial y ciberseguridad en el sector Defensa",
    secciones=[
        ("Introduccion",
         ["La adopcion de inteligencia artificial en el ambito de la ciberdefensa "
          "ha crecido de forma sostenida en la ultima decada, impulsada por la "
          "necesidad de detectar amenazas que evolucionan mas rapido de lo que "
          "los equipos humanos pueden analizar manualmente.",
          "Los ministerios de defensa de varios paises de la region han creado "
          "unidades especializadas que combinan analistas de ciberseguridad con "
          "cientificos de datos, en un esfuerzo por anticipar intrusiones antes "
          "de que comprometan infraestructura critica."]),
        ("Capacidades emergentes",
         ["Los sistemas de deteccion de anomalias basados en aprendizaje "
          "automatico permiten identificar patrones de trafico de red inusuales "
          "en tiempo casi real, lo que reduce significativamente el tiempo de "
          "respuesta ante un incidente.",
          "Ademas, los modelos de clasificacion de malware entrenados sobre "
          "grandes volumenes de muestras facilitan la catalogacion automatica de "
          "nuevas variantes, una tarea que anteriormente dependia casi por "
          "completo del trabajo manual de analistas forenses."]),
        ("Brechas y desafios de talento",
         ["Pese a estos avances, los observatorios de ciencia, tecnologia e "
          "innovacion de la region coinciden en senalar una brecha importante en "
          "la formacion de talento especializado: pocos programas universitarios "
          "combinan de forma explicita ciberseguridad, aprendizaje automatico y "
          "doctrina militar.",
          "Esta brecha limita la capacidad de las fuerzas armadas para retener "
          "personal calificado frente a las ofertas del sector privado, lo que a "
          "su vez condiciona el ritmo de adopcion de estas capacidades "
          "estrategicas."]),
        ("Conclusiones",
         ["El fortalecimiento de capacidades de IA aplicada a la ciberdefensa "
          "requiere una politica publica que articule educacion, investigacion "
          "aplicada y cooperacion regional, de modo que los avances no queden "
          "concentrados unicamente en los paises con mayor presupuesto de "
          "defensa."]),
    ],
))

DOCS.append(dict(
    id="f1_strategic_competition_ai_en",
    fenomeno=1, formato="pdf", idioma="en",
    organizacion="Institute for Strategic Technology Studies (ficticio, estilo RAND/SIPRI)",
    titulo="Strategic Competition in the Age of AI: Risks and Opportunities in Military Applications",
    secciones=[
        ("Executive summary",
         ["Military organizations worldwide are integrating artificial "
          "intelligence into planning, logistics and intelligence analysis at "
          "an accelerating pace. This report summarizes emerging risks and "
          "opportunities associated with that trend, drawing on open-source "
          "policy analysis rather than classified sources.",
          "Four building-block competitions frame the analysis: quantity "
          "versus quality of deployed systems, the balance between hiding and "
          "finding assets on a contested battlefield, centralized versus "
          "decentralized command and control, and the evolving balance between "
          "cyber offense and cyber defense."]),
        ("Quantity versus quality",
         ["Cheaper autonomous systems allow forces to field larger numbers of "
          "expendable platforms, shifting some engagements from a few "
          "high-value assets toward swarms of lower-cost units. This changes "
          "cost-exchange ratios in ways that established doctrine has not fully "
          "absorbed.",
          "At the same time, higher-quality sensor fusion and targeting "
          "algorithms increase the effectiveness of each individual platform, "
          "so quantity and quality improvements can reinforce one another "
          "rather than substitute for each other."]),
        ("Command and control",
         ["Centralized architectures offer tighter coordination but create a "
          "single point of failure if communications are degraded. "
          "Decentralized, AI-assisted decision-making at the edge can preserve "
          "operational tempo under contested conditions, at the cost of "
          "reduced central oversight.",
          "Several allied doctrine documents now explicitly discuss hybrid "
          "architectures that shift authority dynamically depending on the "
          "communications environment, a design pattern borrowed from "
          "commercial distributed systems."]),
        ("Policy recommendations",
         ["The report recommends independent verification and validation "
          "protocols for AI systems prior to fielding, sustained investment in "
          "workforce training, and multilateral dialogue on norms for "
          "AI-enabled targeting to reduce the risk of unintended escalation "
          "between competing powers."]),
    ],
))

DOCS.append(dict(
    id="f1_ia_militar_desafios_pt",
    fenomeno=1, formato="pdf", idioma="pt",
    organizacion="Observatorio Militar de Estudos Estrategicos (ficticio, estilo publicacao militar brasileira)",
    titulo="Inteligencia artificial no contexto militar: desafios eticos e operacionais",
    secciones=[
        ("Introducao",
         ["A incorporacao de tecnicas de inteligencia artificial nas forcas "
          "armadas tem avancado de forma consistente nos ultimos anos, "
          "sobretudo em aplicacoes de manutencao preditiva, analise de imagens "
          "de sensoriamento remoto e apoio ao planejamento de missoes.",
          "Esse processo levanta questoes eticas e operacionais que ainda nao "
          "possuem resposta consolidada na doutrina militar, especialmente no "
          "que se refere ao grau de autonomia permitido a sistemas de apoio a "
          "decisao."]),
        ("Aplicacoes em curso",
         ["Unidades de manutencao de aeronaves ja utilizam modelos preditivos "
          "para antecipar falhas de componentes, reduzindo o tempo de "
          "indisponibilidade das frotas. Da mesma forma, algoritmos de visao "
          "computacional auxiliam na triagem de imagens capturadas por drones "
          "de reconhecimento.",
          "Essas ferramentas nao substituem o analista humano, mas reduzem "
          "significativamente o volume de dados que precisa ser revisado "
          "manualmente antes de uma decisao operacional."]),
        ("Questoes eticas e de governanca",
         ["Entre os principais desafios apontados por especialistas esta o "
          "vies presente em conjuntos de dados de treinamento, que pode "
          "comprometer a confiabilidade dos sistemas em cenarios distintos "
          "daqueles usados no treinamento original.",
          "Tambem se discute a necessidade de manter controle humano "
          "significativo sobre decisoes de maior impacto, de modo a preservar "
          "a responsabilidade institucional e o cumprimento do direito "
          "internacional humanitario."]),
        ("Consideracoes finais",
         ["O fortalecimento responsavel de capacidades de inteligencia "
          "artificial na defesa depende de investimento continuo em formacao "
          "de pessoal especializado, protocolos de teste independentes e "
          "cooperacao com centros academicos de pesquisa aplicada."]),
    ],
))

DOCS.append(dict(
    id="f1_nato_governance_ai_en",
    fenomeno=1, formato="html", idioma="en",
    organizacion="Allied Defense Innovation Review (ficticio, estilo NATO Review)",
    titulo="Allied Approaches to Military AI Governance",
    secciones=[
        ("Why governance matters now",
         ["As member states field an increasing number of AI-enabled systems, "
          "alliance-level governance has become a practical necessity rather "
          "than an abstract policy goal. Divergent national standards for "
          "testing and certification create friction when coalition forces "
          "operate jointly.",
          "A shared baseline for responsible AI use is intended to reduce that "
          "friction without slowing the pace of innovation that member states "
          "consider strategically necessary."]),
        ("Principles under discussion",
         ["Draft principles circulated among defense innovation offices "
          "emphasize traceability of training data, meaningful human "
          "oversight for high-consequence decisions, and mandatory incident "
          "reporting when an AI system behaves unexpectedly during exercises "
          "or operations.",
          "None of these principles are binding treaty obligations; they "
          "function as voluntary commitments that individual member states "
          "translate into their own procurement and testing regulations."]),
        ("Talent and industrial base",
         ["Officials interviewed for this review repeatedly highlighted "
          "workforce shortages as the binding constraint on faster adoption, "
          "more so than the maturity of the underlying technology itself.",
          "Partnerships with universities and dual-use technology firms are "
          "seen as the most promising near-term path to closing that gap, "
          "particularly for smaller member states with limited in-house "
          "research capacity."]),
    ],
))

# ============================= FENOMENO 2 =================================
# Seguridad espacial y orbita baja terrestre (LEO)

DOCS.append(dict(
    id="f2_space_environment_report_en",
    fenomeno=2, formato="pdf", idioma="en",
    organizacion="Orbital Sustainability Office (ficticio, estilo ESA Space Debris Office)",
    titulo="Annual Space Environment Report: Orbital Debris and LEO Congestion",
    secciones=[
        ("Overview of the current environment",
         ["Space surveillance networks now track tens of thousands of objects "
          "larger than ten centimeters in Earth orbit, a small fraction of the "
          "much larger population of debris fragments too small to be "
          "individually cataloged but still capable of damaging operational "
          "spacecraft.",
          "The rate of launches into low Earth orbit has grown sharply in "
          "recent years, driven largely by the deployment of large satellite "
          "constellations for broadband connectivity, which has accelerated "
          "concerns about long-term congestion in the most heavily used "
          "orbital shells."]),
        ("Sources of debris",
         ["Fragmentation events, whether from collisions or from the "
          "break-up of spent rocket stages, remain the dominant source of "
          "trackable debris. Each fragmentation event can generate thousands "
          "of new pieces, many of which persist in orbit for decades.",
          "Intentional destructive testing of satellites has historically "
          "produced some of the largest single increases in tracked debris "
          "population, prompting renewed international calls to restrict such "
          "tests."]),
        ("Mitigation measures",
         ["Post-mission disposal guidelines recommend that satellites in low "
          "Earth orbit re-enter the atmosphere within a bounded number of "
          "years after the end of operations, though compliance across "
          "operators remains uneven.",
          "Active debris removal demonstrations, still at an early "
          "technological readiness level, are being evaluated as a "
          "complementary measure for objects that cannot deorbit "
          "passively within an acceptable timeframe."]),
        ("Outlook for orbital sustainability",
         ["Projections included in this report suggest that without stronger "
          "adherence to mitigation guidelines, the density of debris in the "
          "most congested shells could reach a threshold where collisions "
          "generate debris faster than natural atmospheric drag removes it, a "
          "scenario long referred to in the technical literature as a "
          "cascading instability.",
          "International coordination among space agencies and commercial "
          "operators is identified as the single most important factor in "
          "avoiding that outcome over the coming decade."]),
    ],
))

DOCS.append(dict(
    id="f2_sustainability_challenge_en",
    fenomeno=2, formato="html", idioma="en",
    organizacion="Global Space Policy Forum (ficticio, estilo Secure World Foundation)",
    titulo="Space Sustainability and the Growing Challenge of Orbital Debris",
    secciones=[
        ("A shared but crowded resource",
         ["Outer space is often described as a global commons, yet the most "
          "useful orbital shells are a finite resource that a growing number "
          "of state and commercial actors compete to use safely.",
          "Unlike terrestrial commons, there is no natural process that "
          "quickly restores a congested orbit to a safer state once debris "
          "accumulates, which makes prevention far more cost-effective than "
          "remediation."]),
        ("Policy tools available today",
         ["Voluntary guidelines from international bodies set expectations "
          "for post-mission disposal and collision avoidance maneuvers, but "
          "they generally lack binding enforcement mechanisms, leaving "
          "compliance largely dependent on operator reputation and national "
          "regulation.",
          "Some national regulators have begun to require disposal plans as a "
          "condition of launch licensing, a trend expected to expand as more "
          "countries develop independent launch capacity."]),
        ("The role of transparency",
         ["Publicly available conjunction data, tracking catalogs and "
          "operator disclosures all contribute to a more predictable "
          "operating environment, reducing the likelihood of avoidable "
          "collisions caused by incomplete information.",
          "Civil society organizations and academic groups play an "
          "increasingly visible role in aggregating and interpreting this "
          "data for audiences outside the traditional space industry."]),
    ],
))

DOCS.append(dict(
    id="f2_iadc_status_report_en",
    fenomeno=2, formato="pdf", idioma="en",
    organizacion="Interagency Debris Coordination Panel (ficticio, estilo IADC/UNOOSA)",
    titulo="Status of the Space Debris Environment: Technical Summary",
    secciones=[
        ("Purpose of this summary",
         ["This technical summary consolidates modeling results from several "
          "national space agencies regarding the current and projected state "
          "of the debris environment in low Earth orbit, intended for a "
          "non-specialist policy audience.",
          "It draws on statistical models rather than raw tracking data, and "
          "should be read as an indicative snapshot rather than a real-time "
          "operational assessment."]),
        ("Modeled debris population",
         ["Current models estimate tens of thousands of objects larger than "
          "ten centimeters, more than a million objects between one and ten "
          "centimeters, and well over a hundred million objects smaller than "
          "one centimeter across all orbital regimes.",
          "Objects in the smallest size range are individually untrackable "
          "with current sensor technology but remain capable of causing "
          "mission-ending damage to operational spacecraft at orbital "
          "velocities."]),
        ("Reentry trends",
         ["Intact satellites and rocket bodies now reenter the atmosphere "
          "several times per day on average, reflecting both the growing "
          "population of objects in orbit and improved compliance with "
          "disposal guidelines among newer missions.",
          "Uncontrolled reentries of larger objects continue to raise "
          "questions about ground casualty risk, an area where policy "
          "guidance is still evolving."]),
    ],
))

DOCS.append(dict(
    id="f2_seguridad_espacial_leo_es",
    fenomeno=2, formato="html", idioma="es",
    organizacion="Observatorio Iberoamericano del Espacio (ficticio)",
    titulo="La seguridad del entorno espacial: riesgos de la orbita baja terrestre",
    secciones=[
        ("Un entorno cada vez mas ocupado",
         ["La orbita baja terrestre, esencial para las comunicaciones, la "
          "observacion de la Tierra y la navegacion satelital, enfrenta una "
          "congestion creciente debido al despliegue acelerado de grandes "
          "constelaciones de satelites.",
          "Esta congestion incrementa la probabilidad de colisiones entre "
          "objetos activos y fragmentos de basura espacial, muchos de los "
          "cuales provienen de etapas de cohetes agotadas o de satelites "
          "fuera de servicio."]),
        ("Implicaciones para la region",
         ["Aunque la mayoria de los lanzamientos globales se originan fuera de "
          "America Latina, la region depende cada vez mas de servicios "
          "satelitales para agricultura de precision, monitoreo ambiental y "
          "telecomunicaciones, lo que la hace vulnerable a interrupciones "
          "derivadas de eventos de fragmentacion orbital.",
          "Diversos centros de pensamiento regionales han comenzado a "
          "incorporar la sostenibilidad espacial como un tema de politica "
          "publica, mas alla del ambito estrictamente tecnico de las agencias "
          "espaciales."]),
        ("Medidas de mitigacion",
         ["Las directrices internacionales recomiendan que los satelites en "
          "orbita baja sean retirados de servicio y reingresen a la atmosfera "
          "dentro de un plazo acotado tras el fin de su vida util, aunque el "
          "cumplimiento varia considerablemente entre operadores.",
          "La cooperacion multilateral y el intercambio transparente de datos "
          "de seguimiento orbital se consideran elementos clave para reducir "
          "el riesgo de colisiones evitables."]),
    ],
))

DOCS.append(dict(
    id="f2_sustentabilidade_espacial_pt",
    fenomeno=2, formato="pdf", idioma="pt",
    organizacion="Centro Brasileiro de Estudos Espaciais (ficticio)",
    titulo="Sustentabilidade espacial e gestao de detritos orbitais",
    secciones=[
        ("Contexto geral",
         ["O aumento no numero de lancamentos destinados a orbita baixa da "
          "Terra tem elevado a preocupacao com a sustentabilidade do ambiente "
          "espacial, especialmente apos a implantacao de grandes "
          "constelacoes de satelites de comunicacao.",
          "Agencias espaciais de diversos paises tem investido em modelos "
          "estatisticos para estimar a populacao de detritos e projetar "
          "cenarios futuros de congestionamento orbital."]),
        ("Riscos associados",
         ["Colisoes entre objetos em orbita podem gerar milhares de novos "
          "fragmentos, aumentando exponencialmente o risco para satelites "
          "operacionais nas orbitas mais utilizadas.",
          "Testes destrutivos de satelites, mesmo quando realizados por um "
          "unico pais, tem impacto global sobre a seguranca de todos os "
          "operadores que utilizam a mesma faixa orbital."]),
        ("Cooperacao internacional",
         ["A gestao responsavel do ambiente espacial depende de cooperacao "
          "entre agencias espaciais, operadores comerciais e organismos "
          "multilaterais, incluindo o compartilhamento de dados de "
          "rastreamento e o cumprimento de diretrizes de descarte pos-missao.",
          "Especialistas destacam que a sustentabilidade orbital deve ser "
          "tratada como um bem publico global, dado que nenhum pais pode "
          "garantir sozinho a seguranca do ambiente espacial compartilhado."]),
    ],
))

# ============================= FENOMENO 3 =================================
# Dinamicas territoriales en America Latina

DOCS.append(dict(
    id="f3_panorama_desarrollo_territorial_es",
    fenomeno=3, formato="pdf", idioma="es",
    organizacion="Comision Regional de Desarrollo Territorial (ficticio, estilo CEPAL)",
    titulo="Panorama del desarrollo territorial en America Latina y el Caribe",
    secciones=[
        ("Introduccion",
         ["Las brechas territoriales dentro de los paises de America Latina y "
          "el Caribe siguen siendo, en muchos casos, mas amplias que las "
          "brechas promedio entre paises de la region, lo que plantea "
          "desafios especificos para el diseno de politicas publicas.",
          "Este panorama examina indicadores de ingreso, acceso a servicios "
          "basicos y conectividad a nivel subnacional, con el fin de "
          "identificar patrones territoriales relevantes para la formulacion "
          "de politicas de desarrollo."]),
        ("Desigualdades subnacionales",
         ["Los territorios rurales y las zonas alejadas de los principales "
          "centros urbanos concentran, de manera consistente, los indicadores "
          "mas bajos de acceso a educacion de calidad, salud y "
          "infraestructura digital.",
          "Estas disparidades se acentuan en regiones con mayor presencia de "
          "poblacion indigena o afrodescendiente, lo que sugiere que las "
          "politicas de igualdad territorial deben incorporar de forma "
          "explicita una perspectiva etnica."]),
        ("Migracion y dinamicas territoriales",
         ["Los flujos migratorios internos, motivados por la busqueda de "
          "oportunidades economicas, han transformado la composicion "
          "demografica de varias ciudades intermedias de la region, "
          "generando tanto oportunidades de dinamizacion economica como "
          "presiones sobre servicios publicos locales.",
          "La migracion internacional intrarregional anade una capa adicional "
          "de complejidad, particularmente en zonas fronterizas donde la "
          "capacidad institucional local suele ser limitada."]),
        ("Recomendaciones de politica",
         ["El fortalecimiento de las capacidades fiscales y de planificacion "
          "de los gobiernos subnacionales aparece como una condicion "
          "necesaria para reducir las brechas territoriales de forma "
          "sostenida en el tiempo.",
          "La coordinacion entre niveles de gobierno, junto con sistemas de "
          "informacion territorial mas robustos, permitiria orientar mejor la "
          "inversion publica hacia las zonas con mayores rezagos."]),
    ],
))

DOCS.append(dict(
    id="f3_desarrollo_humano_latam_es",
    fenomeno=3, formato="html", idioma="es",
    organizacion="Programa Regional de Desarrollo Humano (ficticio, estilo PNUD)",
    titulo="Desarrollo humano en America Latina: tendencias y desafios",
    secciones=[
        ("Panorama regional",
         ["America Latina y el Caribe registran avances desiguales en sus "
          "indicadores de desarrollo humano durante la ultima decada, con "
          "mejoras sostenidas en esperanza de vida y anos de escolaridad, "
          "pero con un ingreso nacional bruto per capita que aun no recupera "
          "plenamente los niveles previos a las ultimas crisis economicas "
          "regionales.",
          "La polarizacion politica y la desconfianza institucional se "
          "identifican como factores que condicionan la capacidad de los "
          "Estados para sostener politicas de desarrollo de largo plazo."]),
        ("Percepcion social y confianza institucional",
         ["Encuestas de opinion publica realizadas en distintos paises de la "
          "region muestran niveles de confianza institucional "
          "significativamente mas bajos que en otras regiones del mundo, lo "
          "que se asocia con mayor volatilidad electoral y menor continuidad "
          "de politicas publicas entre gobiernos.",
          "Este fenomeno afecta de forma particular a la juventud, que "
          "reporta consistentemente menores niveles de confianza en las "
          "instituciones democraticas que las generaciones anteriores."]),
        ("Juventud, empleo y educacion",
         ["El desempleo juvenil sigue siendo notablemente mas alto que el "
          "desempleo general en la mayoria de los paises de la region, lo "
          "que refuerza la necesidad de politticas activas de empleo "
          "orientadas especificamente a este grupo etario.",
          "La calidad de la educacion, mas alla del acceso, aparece como un "
          "factor determinante para mejorar las oportunidades laborales de "
          "los jovenes en un mercado de trabajo cada vez mas exigente en "
          "habilidades digitales."]),
    ],
))

DOCS.append(dict(
    id="f3_acuerdo_paz_colombia_es",
    fenomeno=3, formato="pdf", idioma="es",
    organizacion="Instituto de Monitoreo de Acuerdos de Paz (ficticio, estilo Instituto Kroc)",
    titulo="Informe de seguimiento a la implementacion del Acuerdo de Paz en Colombia",
    secciones=[
        ("Objetivo del informe",
         ["Este informe presenta un balance del estado de implementacion de "
          "los compromisos contenidos en el Acuerdo Final de Paz en Colombia, "
          "con enfasis en los puntos relacionados con reforma rural, "
          "participacion politica y garantias de seguridad para "
          "excombatientes y comunidades.",
          "El analisis se basa en fuentes publicas de seguimiento y no "
          "sustituye los informes oficiales de verificacion internacional."]),
        ("Avances identificados",
         ["Se observan avances sostenidos en los compromisos de "
          "reincorporacion economica y social de excombatientes, aunque con "
          "diferencias significativas segun la region del pais donde se "
          "implementan.",
          "Los mecanismos de participacion ciudadana creados por el Acuerdo "
          "han logrado consolidarse en varias zonas priorizadas, si bien su "
          "sostenibilidad financiera de mediano plazo continua siendo un "
          "desafio."]),
        ("Desafios persistentes",
         ["La violencia asociada a economias ilicitas y a la disputa por el "
          "control territorial en zonas de dificil acceso sigue siendo uno de "
          "los principales obstaculos para la implementacion plena del "
          "Acuerdo.",
          "La proteccion de lideres sociales y defensores de derechos "
          "humanos en los territorios mas afectados por el conflicto continua "
          "siendo una prioridad senalada de forma reiterada por organismos de "
          "monitoreo."]),
        ("Conclusiones",
         ["El ritmo de implementacion territorial del Acuerdo varia "
          "sustancialmente entre regiones, lo que subraya la importancia de "
          "una focalizacion diferenciada de la inversion publica y de la "
          "presencia institucional del Estado en los territorios historicamente "
          "mas afectados por el conflicto armado."]),
    ],
))

DOCS.append(dict(
    id="f3_territorial_dynamics_security_en",
    fenomeno=3, formato="html", idioma="en",
    organizacion="Hemispheric Policy Institute (ficticio, estilo think tank regional)",
    titulo="Territorial Dynamics and Human Security in Latin America",
    secciones=[
        ("A regional overview",
         ["Latin America continues to experience uneven territorial "
          "development, with governance capacity, security conditions and "
          "access to public services varying sharply not only between "
          "countries but within them.",
          "These internal disparities often map closely onto historical "
          "patterns of state presence, with peripheral and rural territories "
          "receiving comparatively less sustained institutional investment "
          "than capital regions."]),
        ("Governance and insecurity",
         ["In several countries, non-state armed actors continue to exert de "
          "facto territorial control in areas where formal state institutions "
          "have limited reach, complicating efforts to extend public services "
          "and the rule of law.",
          "This dynamic has direct implications for human security, "
          "particularly for rural communities, indigenous populations and "
          "human rights defenders operating in contested territories."]),
        ("Migration as a territorial phenomenon",
         ["Internal displacement and cross-border migration flows within the "
          "region are closely linked to local security conditions and "
          "economic opportunity, reshaping the demographic profile of both "
          "sending and receiving territories.",
          "Local governments in receiving areas frequently report strained "
          "capacity to absorb these flows, particularly where national "
          "coordination mechanisms remain underdeveloped."]),
        ("Policy implications",
         ["Addressing these territorial dynamics requires coordinated action "
          "across security, social and economic policy domains, rather than "
          "treating each dimension in isolation, since governance gaps and "
          "insecurity tend to reinforce one another over time."]),
    ],
))

DOCS.append(dict(
    id="f3_dinamicas_territoriais_pt",
    fenomeno=3, formato="pdf", idioma="pt",
    organizacion="Nucleo de Estudos Territoriais Latino-Americanos (ficticio)",
    titulo="Dinamicas territoriais e desigualdade na America Latina",
    secciones=[
        ("Introducao",
         ["As desigualdades territoriais na America Latina permanecem entre "
          "as mais persistentes do mundo, refletindo diferencas historicas na "
          "presenca institucional do Estado entre regioes urbanas e rurais.",
          "Este documento discute como essas disparidades se relacionam com "
          "fenomenos como migracao interna, governanca local e acesso a "
          "servicos publicos basicos."]),
        ("Desigualdade e governanca local",
         ["Regioes com menor capacidade fiscal tendem a apresentar indicadores "
          "sociais mais fracos, o que reforca um ciclo de baixo investimento "
          "publico e reduzida capacidade de atrair investimento privado.",
          "O fortalecimento da governanca subnacional e frequentemente "
          "apontado por especialistas como condicao necessaria, ainda que nao "
          "suficiente, para reduzir essas disparidades ao longo do tempo."]),
        ("Migracao e mudanca demografica",
         ["Fluxos migratorios internos, motivados principalmente por busca de "
          "oportunidades economicas, tem alterado a composicao demografica de "
          "varias cidades intermediarias na regiao.",
          "Esses fluxos geram tanto oportunidades de dinamizacao economica "
          "quanto pressoes adicionais sobre a infraestrutura e os servicos "
          "publicos locais, especialmente em areas de fronteira."]),
        ("Consideracoes finais",
         ["Politicas de desenvolvimento territorial mais eficazes exigem "
          "sistemas de informacao subnacional mais robustos e maior "
          "coordenacao entre niveis de governo, de modo a direcionar melhor o "
          "investimento publico para as areas com maiores defasagens."]),
    ],
))


# ---------------------------------------------------------------------------
# Generadores de archivo
# ---------------------------------------------------------------------------

def write_html(doc: dict, out_path: Path) -> None:
    body_sections = []
    for heading, paragraphs in doc["secciones"]:
        paras_html = "\n".join(f"    <p>{html_escape.escape(p)}</p>" for p in paragraphs)
        body_sections.append(f"  <h2>{html_escape.escape(heading)}</h2>\n{paras_html}")
    sections_html = "\n".join(body_sections)

    # Boilerplate de navegacion/pie de pagina deliberado, para probar que el
    # extractor HTML (trafilatura) lo descarta y conserva solo el contenido.
    page = f"""<!doctype html>
<html lang="{doc['idioma']}">
<head><meta charset="utf-8"><title>{html_escape.escape(doc['titulo'])}</title></head>
<body>
<nav>
  <a href="/">Inicio</a> | <a href="/publicaciones">Publicaciones</a> |
  <a href="/sobre-nosotros">Sobre nosotros</a> | <a href="/contacto">Contacto</a>
</nav>
<header>
  <p class="site-name">{html_escape.escape(doc['organizacion'])}</p>
</header>
<article>
  <h1>{html_escape.escape(doc['titulo'])}</h1>
  <p class="byline">Publicado por {html_escape.escape(doc['organizacion'])}</p>
{sections_html}
</article>
<aside>
  <p>Suscribase a nuestro boletin para recibir mas analisis como este.</p>
</aside>
<footer>
  <p>&copy; 2026 {html_escape.escape(doc['organizacion'])}. Todos los derechos reservados.</p>
  <p>Politica de privacidad | Terminos de uso | Mapa del sitio</p>
</footer>
</body>
</html>
"""
    out_path.write_text(page, encoding="utf-8")


def write_pdf(doc: dict, out_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, doc["titulo"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, doc["organizacion"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    for heading, paragraphs in doc["secciones"]:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, heading, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        for p in paragraphs:
            pdf.multi_cell(0, 6, p, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)
        pdf.ln(2)
    pdf.output(str(out_path))


def main() -> None:
    manifest_rows = []
    for doc in DOCS:
        fenomeno_dir = {
            1: "fenomeno_1_ia_defensa",
            2: "fenomeno_2_leo_espacial",
            3: "fenomeno_3_dinamicas_territoriales",
        }[doc["fenomeno"]]
        subdir = "pdf" if doc["formato"] == "pdf" else "html"
        ext = "pdf" if doc["formato"] == "pdf" else "html"
        out_path = CORPUS_DIR / fenomeno_dir / subdir / f"{doc['id']}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if doc["formato"] == "pdf":
            write_pdf(doc, out_path)
        else:
            write_html(doc, out_path)

        n_words = sum(len(p.split()) for _, paras in doc["secciones"] for p in paras)
        manifest_rows.append(
            f"| {doc['id']} | {doc['fenomeno']} | {doc['formato']} | {doc['idioma']} "
            f"| sintetico | {doc['organizacion']} | {doc['titulo']} | ~{n_words} palabras |"
        )
        print(f"generado: {out_path.relative_to(ROOT)}")

    manifest_path = CORPUS_DIR / "fuentes.md"
    header = (
        "# Manifest del corpus de ejemplo (PROVISIONAL)\n\n"
        "**Todos los documentos de este corpus son SINTETICOS**: fueron redactados "
        "a mano por el equipo (`scripts/gen_synthetic_corpus.py`), no descargados. "
        "Esto se debio a que el proxy de salida del entorno de desarrollo bloquea "
        "(403) el acceso a dominios externos como sipri.org, esa.int, cepal.org, "
        "etc. (ver `informe_tecnico.pdf`, seccion de limitaciones). El contenido "
        "esta tematicamente inspirado en fuentes reales conocidas (SIPRI, RAND, "
        "ESA Space Debris Office, UNOOSA/IADC, CEPAL, PNUD, Instituto Kroc, entre "
        "otras) pero **no es una copia ni cita textual de ningun documento real** "
        "y no debe usarse como tal. Este corpus se reemplaza integramente por el "
        "corpus oficial de ADL en cuanto este se encuentre disponible; el pipeline "
        "de `src/` no requiere cambios para procesar el corpus real.\n\n"
        "| id | fenomeno | formato | idioma | tipo | organizacion (ficticia) | titulo | tamano aprox. |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    manifest_path.write_text(header + "\n".join(manifest_rows) + "\n", encoding="utf-8")
    print(f"\nmanifest escrito en: {manifest_path.relative_to(ROOT)}")
    print(f"total documentos generados: {len(DOCS)}")


if __name__ == "__main__":
    main()
