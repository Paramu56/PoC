// 1. Create Category Nodes
CREATE (Welfare:Category {name: "Social Security"})
CREATE (Housing:Category {name: "Housing"})
CREATE (Education:Category {name: "Education & Student Support"})
CREATE (Livelihood:Category {name: "Livelihood & Loans"})

// 2. Create Scheme Nodes
CREATE (YuvaNidhi:Scheme {name: "Yuva Nidhi", type: "Unemployment Stipend"})
CREATE (SSP:Scheme {name: "SSP Scholarship", type: "Academic Stipend"})
CREATE (SandhyaSuraksha:Scheme {name: "Sandhya Suraksha", type: "Old Age Pension"})
CREATE (Maitri:Scheme {name: "Maitri Yojana", type: "Transgender Pension"})
CREATE (PMAY:Scheme {name: "PMAY-G", type: "Housing"})
CREATE (AmbedkarNivasa:Scheme {name: "Ambedkar Nivasa", type: "Housing"})
CREATE (CMEGP:Scheme {name: "CMEGP", type: "Business Loan"})
CREATE (Udyogini:Scheme {name: "Udyogini", type: "Women Business Loan"})
CREATE (Shakti:Scheme {name: "Shakti Yojana", type: "Free Travel"})

// 3. Create Exclusion/Status Nodes
CREATE (StudentStatus:Status {name: "Currently Enrolled Student"})
CREATE (ITPayer:Status {name: "Income Tax Payer"})
CREATE (GovtEmp:Status {name: "Government Employee"})
CREATE (PuccaHouse:Status {name: "Owns Pucca House"})

// 4. Define Dependencies (Green Lines in Graph)
// You need a Ration Card for almost everything
CREATE (YuvaNidhi)-[:DEPENDS_ON]->(BPLCard:Document {name: "BPL/Ration Card"})
CREATE (AmbedkarNivasa)-[:DEPENDS_ON]->(BPLCard)
CREATE (CMEGP)-[:DEPENDS_ON]->(ProjectReport:Document {name: "DPR (Project Report)"})

// 5. Define Exclusions (Red Lines in Graph - The "Hard Stops")
// Mutually Exclusive Pensions
CREATE (SandhyaSuraksha)-[:EXCLUDES]->(Maitri)
CREATE (Maitri)-[:EXCLUDES]->(SandhyaSuraksha)

// Mutually Exclusive Housing
CREATE (PMAY)-[:EXCLUDES]->(AmbedkarNivasa)
CREATE (AmbedkarNivasa)-[:EXCLUDES]->(PMAY)

// Status Based Exclusions
CREATE (StudentStatus)-[:EXCLUDES]->(YuvaNidhi)
CREATE (ITPayer)-[:EXCLUDES]->(GrihaLakshmi:Scheme {name: "Griha Lakshmi"})
CREATE (PuccaHouse)-[:EXCLUDES]->(Housing)
CREATE (GovtEmp)-[:EXCLUDES]->(Udyogini)

// 6. Define Positive Linkages
CREATE (FarmerID:Document {name: "FRUITS FID"})
CREATE (RaithaVidyaNidhi:Scheme {name: "Raitha Vidya Nidhi"})
CREATE (RaithaVidyaNidhi)-[:DEPENDS_ON]->(FarmerID)
CREATE (RaithaVidyaNidhi)-[:COMPLEMENTS]->(SSP) 

RETURN *