<template>
  <div class="search-container">
    <div class="header">
      <img :src="logo" @click="goToHome" style="cursor: pointer;" />
      <SearchInput ref="searchInput" from-search @submit="onSubmit" />
    </div>
    <div class="body">
      <div v-for="(doc, key) in documents" :key="key" class="document-retrieve">
        <div>
          <div class="index">
            <p>{{ key + 1 }}</p>
          </div>
          <div class="text-content">
            <router-link
              :to="{ name: 'document', params: { id: doc.id } }"
              target="_blank"
            >
              Open document
            </router-link>
            <p class="text-preview">{{ doc.text }}</p>
          </div>
        </div>
        <p class="ranking">{{ doc.ranking }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import SearchInput from "./SearchInput";

export default {
  components: {
    SearchInput,
  },
  data() {
    return {
      logo: require("@/assets/logo.png"),
      documents: [],
    };
  },
  mounted() {
    const value = this.$route.query.value;
    this.getDocuments(value);
    this.$refs["searchInput"].setValue(value);
  },
  methods: {
    goToHome() {
      this.$router.push({ name: "home" });
    },
    onSubmit(value) {
      if (value === "") return;
      this.$router.push({ name: "search", query: { value } });
      this.getDocuments(value);
    },
    async getDocuments(value) {
      try {
        const data = await fetch(`http://127.0.0.1:8000/query?value=${value}`);
        let body = JSON.parse(await data.text());
        this.documents = body.map((doc) => {
          return {
            ...doc,
            ranking: doc.ranking.toFixed(5),
          };
        });
      } catch (err) {
        console.log(err);
      }
    },
    openDocument(id) {
      this.$router.push({ name: "document", params: { id } });
    },
  },
};
</script>

<style lang="scss" scoped>
.search-container {
  p {
    margin: 0px;
  }
  .header {
    padding: 25px 20px 20px 20px;
    box-shadow: 0px 0px 3px #b7b7b7;
    display: flex;
    align-items: center;
    > img {
      width: 70px;
      height: 40px;
      border-radius: 5px;
      margin-right: 30px;
    }
  }
  .body {
    display: flex;
    flex-direction: column;
    .document-retrieve {
      display: flex;
      justify-content: space-between;
      padding: 20px 40px 20px 40px;
      border-bottom: 1px solid #ebebeb;
      > div {
        display: flex;
        align-items: center;
        .index {
          background: #353535;
          width: fit-content;
          padding-left: 8px;
          padding-right: 8px;
          height: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 4px;
          p {
            font-weight: 500;
            font-size: 14px;
            color: white;
          }
        }
        .text-content {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding-right: 50px;
          padding-left: 40px;
          .text-preview {
            text-align: left;
            margin-top: 10px;
          }
        }
      }
    }
  }
}
</style>
