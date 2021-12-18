<template>
  <div class="search-container">
    <div class="header">
      <img :src="logo" />
      <SearchInput ref="searchInput" from-search />
    </div>
    <div class="body"></div>
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
      documents: []
    };
  },
  mounted() {
    const value = this.$route.query.value
    this.$refs["searchInput"].setValue(value)
  },
  methods: {
    onSubmit(value) {
      if(value === "") return
      this.getDocuments(value)
    },
    getDocuments(value) {
      // TODO do a request for get documents related to the search value
      console.log(value)
    }
  }
};
</script>

<style lang="scss" scoped>
.search-container {
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
}
</style>
